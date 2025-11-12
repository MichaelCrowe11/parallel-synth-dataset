#!/usr/bin/env python3
"""
Parallel Synth - Image-Text Pair Generation Pipeline
Creates training-ready image-text pairs from rendered samples
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np
from dataclasses import dataclass
import hashlib
from tqdm import tqdm


@dataclass
class ImageTextPair:
    """Represents an image-text training pair"""
    image_path: str
    caption_short: str
    caption_medium: str
    caption_long: str
    caption_technical: str
    caption_artistic: str
    metadata: Dict
    sample_id: str


class ImageTextPairGenerator:
    """Generates image-text pairs from rendered samples"""

    def __init__(self, samples_dir: Path, output_dir: Path):
        self.samples_dir = Path(samples_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pairs = []

    def load_sample(self, sample_dir: Path) -> Optional[ImageTextPair]:
        """Load a sample and create image-text pair"""
        # Find metadata file
        metadata_files = list(sample_dir.glob('*.json'))
        if not metadata_files:
            print(f"Warning: No metadata found in {sample_dir}")
            return None

        metadata_file = metadata_files[0]

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Find image file
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_file = None

        for ext in image_extensions:
            images = list(sample_dir.glob(f'*{ext}'))
            if images:
                image_file = images[0]
                break

        if not image_file:
            print(f"Warning: No image found in {sample_dir}")
            return None

        sample_id = metadata.get('sample_id', sample_dir.name)

        captions = metadata.get('captions', {})

        pair = ImageTextPair(
            image_path=str(image_file),
            caption_short=captions.get('short', ''),
            caption_medium=captions.get('medium', ''),
            caption_long=captions.get('long', ''),
            caption_technical=captions.get('technical', ''),
            caption_artistic=captions.get('artistic', ''),
            metadata=metadata,
            sample_id=sample_id
        )

        return pair

    def process_batch(self) -> List[ImageTextPair]:
        """Process all samples in the directory"""
        sample_dirs = [d for d in self.samples_dir.iterdir() if d.is_dir()]

        print(f"Processing {len(sample_dirs)} samples...")

        for sample_dir in tqdm(sample_dirs, desc="Loading samples"):
            pair = self.load_sample(sample_dir)
            if pair:
                self.pairs.append(pair)

        print(f"✓ Loaded {len(self.pairs)} image-text pairs")
        return self.pairs

    def export_webdataset(self, shard_size: int = 1000):
        """
        Export to WebDataset format for efficient training
        https://github.com/webdataset/webdataset
        """
        try:
            import webdataset as wds
        except ImportError:
            print("Error: webdataset not installed. Install with: pip install webdataset")
            return

        print(f"\nExporting to WebDataset format...")
        print(f"Shard size: {shard_size} samples per shard")

        output_pattern = str(self.output_dir / "parallel-synth-%06d.tar")

        with wds.ShardWriter(output_pattern, maxcount=shard_size) as sink:
            for idx, pair in enumerate(tqdm(self.pairs, desc="Writing shards")):
                # Load image
                image = Image.open(pair.image_path)

                # Convert to bytes
                import io
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()

                # Create sample
                sample = {
                    "__key__": pair.sample_id,
                    "png": img_bytes,
                    "caption_short.txt": pair.caption_short,
                    "caption_medium.txt": pair.caption_medium,
                    "caption_long.txt": pair.caption_long,
                    "caption_technical.txt": pair.caption_technical,
                    "caption_artistic.txt": pair.caption_artistic,
                    "metadata.json": json.dumps(pair.metadata),
                }

                sink.write(sample)

        print(f"✓ WebDataset export complete")

    def export_jsonl(self, caption_type: str = 'medium'):
        """Export to JSONL format (one JSON per line)"""
        output_file = self.output_dir / f"parallel_synth_{caption_type}.jsonl"

        print(f"\nExporting to JSONL format...")
        print(f"Caption type: {caption_type}")
        print(f"Output: {output_file}")

        with open(output_file, 'w') as f:
            for pair in tqdm(self.pairs, desc="Writing JSONL"):
                caption = getattr(pair, f'caption_{caption_type}', pair.caption_medium)

                record = {
                    'image_path': pair.image_path,
                    'caption': caption,
                    'sample_id': pair.sample_id,
                    'metadata': pair.metadata
                }

                f.write(json.dumps(record) + '\n')

        print(f"✓ JSONL export complete: {len(self.pairs)} records")

    def export_parquet(self):
        """Export to Parquet format for efficient storage and querying"""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            print("Error: pyarrow not installed. Install with: pip install pyarrow")
            return

        print(f"\nExporting to Parquet format...")

        data = {
            'sample_id': [],
            'image_path': [],
            'caption_short': [],
            'caption_medium': [],
            'caption_long': [],
            'caption_technical': [],
            'caption_artistic': [],
            'metadata_json': []
        }

        for pair in self.pairs:
            data['sample_id'].append(pair.sample_id)
            data['image_path'].append(pair.image_path)
            data['caption_short'].append(pair.caption_short)
            data['caption_medium'].append(pair.caption_medium)
            data['caption_long'].append(pair.caption_long)
            data['caption_technical'].append(pair.caption_technical)
            data['caption_artistic'].append(pair.caption_artistic)
            data['metadata_json'].append(json.dumps(pair.metadata))

        table = pa.table(data)

        output_file = self.output_dir / "parallel_synth_dataset.parquet"
        pq.write_table(table, output_file, compression='snappy')

        print(f"✓ Parquet export complete: {output_file}")
        print(f"  Rows: {len(self.pairs)}")
        print(f"  Size: {output_file.stat().st_size / (1024*1024):.2f} MB")

    def export_csv(self, caption_type: str = 'medium'):
        """Export to CSV format"""
        import csv

        output_file = self.output_dir / f"parallel_synth_{caption_type}.csv"

        print(f"\nExporting to CSV format...")

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'image_path', 'caption'])

            for pair in tqdm(self.pairs, desc="Writing CSV"):
                caption = getattr(pair, f'caption_{caption_type}', pair.caption_medium)
                writer.writerow([pair.sample_id, pair.image_path, caption])

        print(f"✓ CSV export complete: {output_file}")

    def create_train_val_split(self, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Create train/validation/test splits"""
        import random

        print(f"\nCreating train/val/test split...")
        print(f"Validation ratio: {val_ratio}")
        print(f"Test ratio: {test_ratio}")

        total = len(self.pairs)
        indices = list(range(total))
        random.shuffle(indices)

        val_size = int(total * val_ratio)
        test_size = int(total * test_ratio)
        train_size = total - val_size - test_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        splits = {
            'train': [self.pairs[i] for i in train_indices],
            'val': [self.pairs[i] for i in val_indices],
            'test': [self.pairs[i] for i in test_indices]
        }

        print(f"✓ Split created:")
        print(f"  Train: {len(splits['train'])} samples")
        print(f"  Val: {len(splits['val'])} samples")
        print(f"  Test: {len(splits['test'])} samples")

        # Save split indices
        split_file = self.output_dir / "split_indices.json"
        with open(split_file, 'w') as f:
            json.dump({
                'train': train_indices,
                'val': val_indices,
                'test': test_indices
            }, f, indent=2)

        print(f"✓ Split indices saved: {split_file}")

        return splits

    def generate_statistics(self) -> Dict:
        """Generate dataset statistics"""
        print(f"\nGenerating statistics...")

        stats = {
            'total_samples': len(self.pairs),
            'categories': {},
            'caption_lengths': {
                'short': [],
                'medium': [],
                'long': []
            }
        }

        for pair in self.pairs:
            # Caption lengths
            stats['caption_lengths']['short'].append(len(pair.caption_short.split()))
            stats['caption_lengths']['medium'].append(len(pair.caption_medium.split()))
            stats['caption_lengths']['long'].append(len(pair.caption_long.split()))

            # Category counts
            if 'categories' in pair.metadata:
                categories = pair.metadata['categories']
                for category in categories.keys():
                    if category not in stats['categories']:
                        stats['categories'][category] = 0
                    stats['categories'][category] += 1

        # Calculate averages
        for caption_type in ['short', 'medium', 'long']:
            lengths = stats['caption_lengths'][caption_type]
            stats['caption_lengths'][caption_type] = {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths)
            }

        # Save statistics
        stats_file = self.output_dir / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"✓ Statistics saved: {stats_file}")

        # Print summary
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Categories:")
        for category, count in sorted(stats['categories'].items()):
            print(f"    {category}: {count}")
        print(f"  Caption lengths (mean words):")
        for caption_type, length_stats in stats['caption_lengths'].items():
            print(f"    {caption_type}: {length_stats['mean']:.1f} ± {length_stats['std']:.1f}")

        return stats


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Parallel Synth Image-Text Pair Generator')
    parser.add_argument('--samples-dir', type=str, required=True, help='Directory containing samples')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--format', choices=['webdataset', 'jsonl', 'parquet', 'csv', 'all'],
                       default='all', help='Export format')
    parser.add_argument('--caption-type', choices=['short', 'medium', 'long', 'technical', 'artistic'],
                       default='medium', help='Caption type for single-caption formats')
    parser.add_argument('--shard-size', type=int, default=1000, help='Samples per shard for WebDataset')
    parser.add_argument('--split', action='store_true', help='Create train/val/test split')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test ratio')

    args = parser.parse_args()

    generator = ImageTextPairGenerator(Path(args.samples_dir), Path(args.output_dir))

    # Load samples
    generator.process_batch()

    if not generator.pairs:
        print("Error: No valid samples found")
        return 1

    # Export in requested format(s)
    if args.format in ['webdataset', 'all']:
        generator.export_webdataset(args.shard_size)

    if args.format in ['jsonl', 'all']:
        generator.export_jsonl(args.caption_type)

    if args.format in ['parquet', 'all']:
        generator.export_parquet()

    if args.format in ['csv', 'all']:
        generator.export_csv(args.caption_type)

    # Create split if requested
    if args.split:
        generator.create_train_val_split(args.val_ratio, args.test_ratio)

    # Generate statistics
    generator.generate_statistics()

    print(f"\n✓ Pipeline complete!")

    return 0


if __name__ == '__main__':
    exit(main())
