#!/usr/bin/env python3
"""
Parallel Synth - PyTorch DataLoader
Efficient data loading for training multimodal models
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms as transforms
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import random


class ParallelSynthDataset(Dataset):
    """PyTorch Dataset for Parallel Synth data"""

    def __init__(
        self,
        data_dir: Path,
        caption_type: str = 'medium',
        transform: Optional[Callable] = None,
        load_metadata: bool = False,
        max_samples: Optional[int] = None
    ):
        """
        Initialize dataset

        Args:
            data_dir: Directory containing samples
            caption_type: Which caption to use (short/medium/long/technical/artistic)
            transform: Image transformations
            load_metadata: Whether to load full metadata
            max_samples: Maximum number of samples to load
        """
        self.data_dir = Path(data_dir)
        self.caption_type = caption_type
        self.transform = transform
        self.load_metadata = load_metadata

        # Collect all samples
        self.samples = self._collect_samples()

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def _collect_samples(self) -> List[Dict]:
        """Collect all samples from directory"""
        samples = []

        for sample_dir in self.data_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            # Find metadata file
            metadata_files = list(sample_dir.glob('*.json'))
            if not metadata_files:
                continue

            # Find image file
            image_files = list(sample_dir.glob('*.png')) + list(sample_dir.glob('*.jpg'))
            if not image_files:
                continue

            samples.append({
                'sample_id': sample_dir.name,
                'image_path': image_files[0],
                'metadata_path': metadata_files[0]
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item"""
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Load metadata
        with open(sample['metadata_path'], 'r') as f:
            metadata = json.load(f)

        # Get caption
        captions = metadata.get('captions', {})
        caption = captions.get(self.caption_type, captions.get('medium', ''))

        result = {
            'image': image,
            'caption': caption,
            'sample_id': sample['sample_id']
        }

        if self.load_metadata:
            result['metadata'] = metadata

        return result


class ParallelSynthWebDataset(IterableDataset):
    """WebDataset-based iterable dataset for large-scale training"""

    def __init__(
        self,
        shard_pattern: str,
        caption_type: str = 'medium',
        transform: Optional[Callable] = None,
        shuffle_buffer: int = 1000
    ):
        """
        Initialize WebDataset

        Args:
            shard_pattern: Pattern for shard files (e.g., "data/parallel-synth-{000000..000099}.tar")
            caption_type: Which caption to use
            transform: Image transformations
            shuffle_buffer: Shuffle buffer size
        """
        try:
            import webdataset as wds
        except ImportError:
            raise ImportError("webdataset not installed. Install with: pip install webdataset")

        self.shard_pattern = shard_pattern
        self.caption_type = caption_type
        self.transform = transform
        self.shuffle_buffer = shuffle_buffer

        # Create WebDataset pipeline
        self.dataset = (
            wds.WebDataset(shard_pattern)
            .shuffle(shuffle_buffer)
            .decode("pil")
            .to_tuple("png", f"caption_{caption_type}.txt")
            .map(self._process_sample)
        )

    def _process_sample(self, sample: Tuple) -> Dict[str, torch.Tensor]:
        """Process a single sample"""
        image, caption = sample

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'caption': caption
        }

    def __iter__(self):
        return iter(self.dataset)


class ParallelSynthJSONLDataset(Dataset):
    """Dataset that loads from JSONL format"""

    def __init__(
        self,
        jsonl_path: Path,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize JSONL dataset

        Args:
            jsonl_path: Path to JSONL file
            transform: Image transformations
            max_samples: Maximum number of samples
        """
        self.jsonl_path = Path(jsonl_path)
        self.transform = transform

        # Load all records
        self.records = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.records.append(json.loads(line))
                if max_samples and len(self.records) >= max_samples:
                    break

        print(f"Loaded {len(self.records)} records from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item"""
        record = self.records[idx]

        # Load image
        image = Image.open(record['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'caption': record['caption'],
            'sample_id': record.get('sample_id', '')
        }


def get_default_transforms(image_size: int = 512, augment: bool = False) -> transforms.Compose:
    """Get default image transformations"""
    transform_list = []

    if augment:
        transform_list.extend([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])
    else:
        transform_list.extend([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size)
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transforms.Compose(transform_list)


def create_dataloader(
    dataset_type: str,
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 512,
    augment: bool = False,
    caption_type: str = 'medium',
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for training

    Args:
        dataset_type: Type of dataset ('directory', 'webdataset', 'jsonl')
        data_path: Path to data
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Image size
        augment: Whether to use augmentation
        caption_type: Caption type to use
        shuffle: Whether to shuffle
        **kwargs: Additional arguments for dataset

    Returns:
        PyTorch DataLoader
    """
    transform = get_default_transforms(image_size, augment)

    if dataset_type == 'directory':
        dataset = ParallelSynthDataset(
            Path(data_path),
            caption_type=caption_type,
            transform=transform,
            **kwargs
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    elif dataset_type == 'webdataset':
        dataset = ParallelSynthWebDataset(
            data_path,
            caption_type=caption_type,
            transform=transform,
            **kwargs
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )

    elif dataset_type == 'jsonl':
        dataset = ParallelSynthJSONLDataset(
            Path(data_path),
            transform=transform,
            **kwargs
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batch processing"""
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    sample_ids = [item.get('sample_id', '') for item in batch]

    return {
        'images': images,
        'captions': captions,
        'sample_ids': sample_ids
    }


# Example usage
if __name__ == '__main__':
    # Example: Create dataloader for training
    train_loader = create_dataloader(
        dataset_type='directory',
        data_path='./output/samples',
        batch_size=16,
        num_workers=4,
        image_size=512,
        augment=True,
        caption_type='medium'
    )

    print(f"DataLoader created with {len(train_loader)} batches")

    # Test batch
    for batch in train_loader:
        print(f"Batch shape: {batch['image'].shape}")
        print(f"Number of captions: {len(batch['caption'])}")
        print(f"Sample caption: {batch['caption'][0]}")
        break
