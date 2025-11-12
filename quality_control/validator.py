#!/usr/bin/env python3
"""
Parallel Synth - Quality Control & Validation
Validates generated samples for quality and correctness
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import cv2
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class ValidationResult:
    """Result of sample validation"""
    sample_id: str
    is_valid: bool
    quality_score: float
    issues: List[str]
    metrics: Dict[str, float]


class DatasetValidator:
    """Validates dataset samples for quality"""

    def __init__(self, min_resolution: Tuple[int, int] = (512, 512)):
        self.min_resolution = min_resolution
        self.validation_results = []

    def validate_sample(self, sample_dir: Path) -> ValidationResult:
        """Validate a single sample"""
        issues = []
        metrics = {}

        # Find metadata and image
        metadata_files = list(sample_dir.glob('*.json'))
        image_files = list(sample_dir.glob('*.png')) + list(sample_dir.glob('*.jpg'))

        sample_id = sample_dir.name

        if not metadata_files:
            issues.append("Missing metadata file")
            return ValidationResult(sample_id, False, 0.0, issues, metrics)

        if not image_files:
            issues.append("Missing image file")
            return ValidationResult(sample_id, False, 0.0, issues, metrics)

        # Load metadata
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)

        # Load image
        image = Image.open(image_files[0])
        img_array = np.array(image)

        # Validation checks
        issues.extend(self.check_metadata(metadata))
        issues.extend(self.check_image_quality(img_array))
        issues.extend(self.check_captions(metadata.get('captions', {})))

        # Calculate metrics
        metrics = self.calculate_metrics(img_array, metadata)

        # Calculate quality score
        quality_score = self.calculate_quality_score(metrics, issues)

        is_valid = len(issues) == 0 and quality_score >= 0.7

        return ValidationResult(sample_id, is_valid, quality_score, issues, metrics)

    def check_metadata(self, metadata: Dict) -> List[str]:
        """Check metadata completeness and validity"""
        issues = []

        required_fields = ['sample_id', 'version', 'timestamp', 'categories']

        for field in required_fields:
            if field not in metadata:
                issues.append(f"Missing required field: {field}")

        # Check categories
        if 'categories' in metadata:
            if not metadata['categories']:
                issues.append("Categories dict is empty")

        # Check captions
        if 'captions' in metadata:
            captions = metadata['captions']
            if not captions.get('short') and not captions.get('medium'):
                issues.append("Missing required captions")

        return issues

    def check_image_quality(self, img_array: np.ndarray) -> List[str]:
        """Check image quality metrics"""
        issues = []

        # Check resolution
        height, width = img_array.shape[:2]
        if height < self.min_resolution[1] or width < self.min_resolution[0]:
            issues.append(f"Resolution {width}x{height} below minimum {self.min_resolution}")

        # Check if image is completely black
        if np.mean(img_array) < 1.0:
            issues.append("Image is completely black")

        # Check if image is completely white
        if np.mean(img_array) > 254.0:
            issues.append("Image is completely white")

        # Check for NaN or Inf values
        if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
            issues.append("Image contains NaN or Inf values")

        # Check color channels
        if len(img_array.shape) == 2:
            # Grayscale
            pass
        elif len(img_array.shape) == 3:
            if img_array.shape[2] not in [3, 4]:
                issues.append(f"Unexpected number of color channels: {img_array.shape[2]}")
        else:
            issues.append(f"Invalid image shape: {img_array.shape}")

        return issues

    def check_captions(self, captions: Dict) -> List[str]:
        """Check caption quality"""
        issues = []

        if not captions:
            issues.append("No captions provided")
            return issues

        # Check caption lengths
        if 'short' in captions:
            word_count = len(captions['short'].split())
            if word_count < 3:
                issues.append(f"Short caption too brief: {word_count} words")
            elif word_count > 50:
                issues.append(f"Short caption too long: {word_count} words")

        if 'medium' in captions:
            word_count = len(captions['medium'].split())
            if word_count < 10:
                issues.append(f"Medium caption too brief: {word_count} words")
            elif word_count > 150:
                issues.append(f"Medium caption too long: {word_count} words")

        # Check for empty captions
        for caption_type, caption_text in captions.items():
            if not caption_text.strip():
                issues.append(f"Empty {caption_type} caption")

        return issues

    def calculate_metrics(self, img_array: np.ndarray, metadata: Dict) -> Dict[str, float]:
        """Calculate quality metrics"""
        metrics = {}

        # Brightness
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        metrics['brightness'] = float(np.mean(gray))
        metrics['contrast'] = float(np.std(gray))

        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics['sharpness'] = float(laplacian.var())

        # Color variance
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            metrics['color_variance'] = float(np.var(img_array))

        # Metadata completeness score
        total_fields = 10  # Expected number of metadata fields
        present_fields = len(metadata.keys())
        metrics['metadata_completeness'] = min(1.0, present_fields / total_fields)

        # Caption completeness
        caption_count = len(metadata.get('captions', {}))
        expected_captions = 5  # short, medium, long, technical, artistic
        metrics['caption_completeness'] = min(1.0, caption_count / expected_captions)

        return metrics

    def calculate_quality_score(self, metrics: Dict[str, float], issues: List[str]) -> float:
        """Calculate overall quality score (0-1)"""
        score = 1.0

        # Penalize for issues
        score -= len(issues) * 0.1

        # Brightness score (should be in reasonable range)
        brightness = metrics.get('brightness', 128)
        if brightness < 20 or brightness > 235:
            score -= 0.2

        # Contrast score (should have decent contrast)
        contrast = metrics.get('contrast', 0)
        if contrast < 10:
            score -= 0.2

        # Sharpness score
        sharpness = metrics.get('sharpness', 0)
        if sharpness < 100:
            score -= 0.1

        # Metadata completeness
        metadata_score = metrics.get('metadata_completeness', 0)
        score *= metadata_score

        # Caption completeness
        caption_score = metrics.get('caption_completeness', 0)
        score *= caption_score

        return max(0.0, min(1.0, score))

    def validate_batch(self, samples_dir: Path) -> List[ValidationResult]:
        """Validate all samples in directory"""
        sample_dirs = [d for d in samples_dir.iterdir() if d.is_dir()]

        print(f"\nValidating {len(sample_dirs)} samples...")

        for sample_dir in tqdm(sample_dirs, desc="Validating"):
            result = self.validate_sample(sample_dir)
            self.validation_results.append(result)

        return self.validation_results

    def generate_report(self, output_path: Path):
        """Generate validation report"""
        if not self.validation_results:
            print("No validation results to report")
            return

        total = len(self.validation_results)
        valid = sum(1 for r in self.validation_results if r.is_valid)
        invalid = total - valid

        avg_quality = np.mean([r.quality_score for r in self.validation_results])

        # Collect all issues
        issue_counts = {}
        for result in self.validation_results:
            for issue in result.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        # Quality distribution
        quality_bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        quality_hist = np.histogram([r.quality_score for r in self.validation_results], bins=quality_bins)[0]

        report = {
            'summary': {
                'total_samples': total,
                'valid_samples': valid,
                'invalid_samples': invalid,
                'validation_rate': valid / total if total > 0 else 0,
                'average_quality_score': float(avg_quality)
            },
            'quality_distribution': {
                f"{quality_bins[i]:.1f}-{quality_bins[i+1]:.1f}": int(count)
                for i, count in enumerate(quality_hist)
            },
            'common_issues': sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'failed_samples': [
                {
                    'sample_id': r.sample_id,
                    'quality_score': r.quality_score,
                    'issues': r.issues
                }
                for r in self.validation_results if not r.is_valid
            ][:100]  # Limit to first 100
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*60}")
        print("Validation Report")
        print(f"{'='*60}")
        print(f"\nTotal samples: {total}")
        print(f"Valid: {valid} ({valid/total*100:.1f}%)")
        print(f"Invalid: {invalid} ({invalid/total*100:.1f}%)")
        print(f"Average quality score: {avg_quality:.3f}")

        print(f"\nQuality distribution:")
        for range_str, count in report['quality_distribution'].items():
            percentage = count / total * 100 if total > 0 else 0
            print(f"  {range_str}: {count} ({percentage:.1f}%)")

        print(f"\nMost common issues:")
        for issue, count in report['common_issues'][:5]:
            print(f"  {issue}: {count}")

        print(f"\n✓ Report saved to {output_path}")

    def filter_valid_samples(self, samples_dir: Path, output_dir: Path, min_quality: float = 0.7):
        """Copy only valid samples to output directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        valid_results = [r for r in self.validation_results if r.is_valid and r.quality_score >= min_quality]

        print(f"\nFiltering valid samples (min quality: {min_quality})...")
        print(f"Found {len(valid_results)} valid samples")

        import shutil

        for result in tqdm(valid_results, desc="Copying"):
            src_dir = samples_dir / result.sample_id
            dst_dir = output_dir / result.sample_id

            if src_dir.exists():
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        print(f"✓ Valid samples copied to {output_dir}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Parallel Synth Dataset Validator')
    parser.add_argument('--samples-dir', type=str, required=True, help='Directory containing samples')
    parser.add_argument('--output-dir', type=str, help='Output directory for filtered samples')
    parser.add_argument('--report', type=str, default='validation_report.json', help='Report output path')
    parser.add_argument('--min-quality', type=float, default=0.7, help='Minimum quality score for filtering')
    parser.add_argument('--min-resolution', type=int, nargs=2, default=[512, 512],
                       help='Minimum resolution (width height)')

    args = parser.parse_args()

    validator = DatasetValidator(min_resolution=tuple(args.min_resolution))

    # Validate all samples
    validator.validate_batch(Path(args.samples_dir))

    # Generate report
    validator.generate_report(Path(args.report))

    # Filter valid samples if output directory specified
    if args.output_dir:
        validator.filter_valid_samples(
            Path(args.samples_dir),
            Path(args.output_dir),
            args.min_quality
        )

    return 0


if __name__ == '__main__':
    exit(main())
