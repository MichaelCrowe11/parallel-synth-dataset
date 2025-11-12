#!/usr/bin/env python3
"""
Parallel Synth - Video Frame Extraction & Annotation
Extracts frames from CGI films and creates annotations
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
from datetime import timedelta
from tqdm import tqdm
import uuid


@dataclass
class FrameAnnotation:
    """Annotation for a single frame"""
    frame_number: int
    timestamp: float
    timestamp_str: str
    shot_id: str
    scene_id: str
    camera_movement: Optional[str]
    lighting_type: Optional[str]
    dominant_colors: List[str]
    objects: List[str]
    caption: str
    technical_notes: str


class VideoFrameExtractor:
    """Extracts and annotates frames from video files"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)

        self.annotations = []

    def extract_frames(
        self,
        video_path: Path,
        fps_target: Optional[float] = None,
        every_nth_frame: int = 1,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        quality: int = 95
    ) -> List[Dict]:
        """
        Extract frames from video

        Args:
            video_path: Path to video file
            fps_target: Target FPS (if None, uses source FPS)
            every_nth_frame: Extract every Nth frame
            start_time: Start time in seconds
            end_time: End time in seconds (None = end of video)
            quality: JPEG quality (0-100)

        Returns:
            List of extracted frame metadata
        """
        print(f"\nExtracting frames from: {video_path.name}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Duration: {duration:.2f}s ({total_frames} frames)")

        # Calculate frame extraction parameters
        if end_time is None:
            end_time = duration

        start_frame = int(start_time * fps)
        end_frame = int(min(end_time * fps, total_frames))

        if fps_target:
            frame_interval = int(fps / fps_target)
        else:
            frame_interval = every_nth_frame

        print(f"\nExtraction settings:")
        print(f"  Start: {start_time}s (frame {start_frame})")
        print(f"  End: {end_time}s (frame {end_frame})")
        print(f"  Frame interval: {frame_interval}")

        # Extract frames
        extracted_frames = []
        frame_count = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        pbar = tqdm(total=end_frame - start_frame, desc="Extracting frames")

        while cap.isOpened():
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if frame_number >= end_frame:
                break

            ret, frame = cap.read()

            if not ret:
                break

            if (frame_number - start_frame) % frame_interval == 0:
                # Generate frame ID
                frame_id = f"{video_path.stem}_{frame_number:08d}"

                # Save frame
                frame_path = self.frames_dir / f"{frame_id}.jpg"
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

                # Calculate timestamp
                timestamp = frame_number / fps
                timestamp_str = str(timedelta(seconds=timestamp))

                # Analyze frame
                dominant_colors = self.extract_dominant_colors(frame)

                # Create metadata
                frame_metadata = {
                    'frame_id': frame_id,
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'timestamp_str': timestamp_str,
                    'video_source': video_path.name,
                    'resolution': {'width': width, 'height': height},
                    'dominant_colors': dominant_colors,
                    'file_path': str(frame_path)
                }

                extracted_frames.append(frame_metadata)
                frame_count += 1

            pbar.update(1)

        pbar.close()
        cap.release()

        print(f"✓ Extracted {frame_count} frames")

        return extracted_frames

    def extract_dominant_colors(self, frame: np.ndarray, num_colors: int = 5) -> List[str]:
        """Extract dominant colors from frame using k-means"""
        # Resize for faster processing
        small_frame = cv2.resize(frame, (150, 150))

        # Convert to RGB
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Reshape to list of pixels
        pixels = rgb_frame.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert to hex colors
        hex_colors = []
        for center in centers:
            r, g, b = center.astype(int)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            hex_colors.append(hex_color)

        return hex_colors

    def detect_shot_boundaries(self, video_path: Path, threshold: float = 30.0) -> List[int]:
        """
        Detect shot boundaries in video using frame difference

        Args:
            video_path: Path to video
            threshold: Difference threshold for shot detection

        Returns:
            List of frame numbers where shots change
        """
        print(f"\nDetecting shot boundaries in: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        shot_boundaries = [0]  # First frame is always a boundary
        prev_frame = None

        for frame_num in tqdm(range(total_frames), desc="Analyzing frames"):
            ret, frame = cap.read()

            if not ret:
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)

                if mean_diff > threshold:
                    shot_boundaries.append(frame_num)

            prev_frame = gray

        cap.release()

        print(f"✓ Detected {len(shot_boundaries)} shots")

        return shot_boundaries

    def annotate_frames_batch(
        self,
        frames_metadata: List[Dict],
        annotation_model: Optional[str] = None
    ) -> List[FrameAnnotation]:
        """
        Generate annotations for extracted frames

        Args:
            frames_metadata: List of frame metadata dicts
            annotation_model: Model to use for annotation (future: CLIP, BLIP, etc.)

        Returns:
            List of frame annotations
        """
        print(f"\nAnnotating {len(frames_metadata)} frames...")

        annotations = []

        for frame_meta in tqdm(frames_metadata, desc="Generating annotations"):
            # Basic annotation (can be enhanced with ML models)
            annotation = FrameAnnotation(
                frame_number=frame_meta['frame_number'],
                timestamp=frame_meta['timestamp'],
                timestamp_str=frame_meta['timestamp_str'],
                shot_id=f"shot_{frame_meta['frame_number'] // 100}",  # Simple shot grouping
                scene_id=f"scene_{frame_meta['frame_number'] // 500}",  # Simple scene grouping
                camera_movement=None,  # To be filled by ML model
                lighting_type=self.classify_lighting(frame_meta['dominant_colors']),
                dominant_colors=frame_meta['dominant_colors'],
                objects=[],  # To be filled by object detection
                caption=self.generate_caption(frame_meta),
                technical_notes=json.dumps(frame_meta, indent=2)
            )

            annotations.append(annotation)

        print(f"✓ Generated {len(annotations)} annotations")

        return annotations

    def classify_lighting(self, dominant_colors: List[str]) -> str:
        """Classify lighting type based on dominant colors"""
        # Convert hex to RGB
        rgb_colors = []
        for hex_color in dominant_colors:
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            rgb_colors.append((r, g, b))

        # Calculate average brightness
        avg_brightness = np.mean([sum(rgb) / 3 for rgb in rgb_colors])

        if avg_brightness > 180:
            return "high_key"
        elif avg_brightness < 80:
            return "low_key"
        else:
            return "medium"

    def generate_caption(self, frame_meta: Dict) -> str:
        """Generate basic caption for frame"""
        timestamp = frame_meta['timestamp_str']
        colors = ', '.join(frame_meta['dominant_colors'][:3])

        return f"Frame at {timestamp} with dominant colors: {colors}"

    def export_annotations(self, annotations: List[FrameAnnotation]):
        """Export annotations to JSON"""
        annotations_file = self.output_dir / "annotations.json"

        annotations_data = []
        for ann in annotations:
            annotations_data.append({
                'frame_number': ann.frame_number,
                'timestamp': ann.timestamp,
                'timestamp_str': ann.timestamp_str,
                'shot_id': ann.shot_id,
                'scene_id': ann.scene_id,
                'camera_movement': ann.camera_movement,
                'lighting_type': ann.lighting_type,
                'dominant_colors': ann.dominant_colors,
                'objects': ann.objects,
                'caption': ann.caption,
                'technical_notes': ann.technical_notes
            })

        with open(annotations_file, 'w') as f:
            json.dump(annotations_data, f, indent=2)

        print(f"✓ Annotations saved: {annotations_file}")

    def create_training_dataset(self, annotations: List[FrameAnnotation]):
        """Create training dataset from annotations"""
        dataset_file = self.output_dir / "training_dataset.jsonl"

        print(f"\nCreating training dataset...")

        with open(dataset_file, 'w') as f:
            for ann in annotations:
                frame_path = self.frames_dir / f"{ann.shot_id}_{ann.frame_number:08d}.jpg"

                record = {
                    'image_path': str(frame_path),
                    'caption': ann.caption,
                    'timestamp': ann.timestamp,
                    'lighting': ann.lighting_type,
                    'colors': ann.dominant_colors,
                    'shot_id': ann.shot_id,
                    'scene_id': ann.scene_id
                }

                f.write(json.dumps(record) + '\n')

        print(f"✓ Training dataset saved: {dataset_file}")
        print(f"  Samples: {len(annotations)}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Parallel Synth Video Frame Extractor')
    parser.add_argument('--video', type=str, required=True, help='Input video file')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--fps', type=float, help='Target FPS (default: source FPS)')
    parser.add_argument('--every-nth', type=int, default=1, help='Extract every Nth frame')
    parser.add_argument('--start', type=float, default=0.0, help='Start time in seconds')
    parser.add_argument('--end', type=float, help='End time in seconds')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality (0-100)')
    parser.add_argument('--detect-shots', action='store_true', help='Detect shot boundaries')
    parser.add_argument('--shot-threshold', type=float, default=30.0, help='Shot detection threshold')
    parser.add_argument('--annotate', action='store_true', help='Generate annotations')

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    extractor = VideoFrameExtractor(Path(args.output_dir))

    # Detect shot boundaries if requested
    if args.detect_shots:
        shot_boundaries = extractor.detect_shot_boundaries(video_path, args.shot_threshold)
        shots_file = extractor.output_dir / "shot_boundaries.json"
        with open(shots_file, 'w') as f:
            json.dump({'shots': shot_boundaries}, f, indent=2)
        print(f"✓ Shot boundaries saved: {shots_file}")

    # Extract frames
    frames = extractor.extract_frames(
        video_path,
        fps_target=args.fps,
        every_nth_frame=args.every_nth,
        start_time=args.start,
        end_time=args.end,
        quality=args.quality
    )

    # Save frame metadata
    frames_file = extractor.output_dir / "frames_metadata.json"
    with open(frames_file, 'w') as f:
        json.dump({'frames': frames}, f, indent=2)
    print(f"✓ Frame metadata saved: {frames_file}")

    # Annotate if requested
    if args.annotate:
        annotations = extractor.annotate_frames_batch(frames)
        extractor.export_annotations(annotations)
        extractor.create_training_dataset(annotations)

    print(f"\n✓ Complete!")

    return 0


if __name__ == '__main__':
    exit(main())
