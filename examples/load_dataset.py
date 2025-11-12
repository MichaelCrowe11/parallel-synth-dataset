#!/usr/bin/env python3
"""
Example: Load and Use Parallel Synth Dataset

This example shows how to:
1. Load the dataset with PyTorch
2. Visualize samples
3. Use in training loops
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataloader import create_dataloader
import matplotlib.pyplot as plt
import numpy as np

def visualize_batch(dataloader, num_samples=4):
    """Visualize samples from the dataset"""
    print("Loading batch...")

    # Get one batch
    batch = next(iter(dataloader))

    images = batch['images'].numpy()
    captions = batch['captions']

    # Denormalize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(min(num_samples, len(images))):
        img = images[i].transpose(1, 2, 0)  # CHW -> HWC
        img = img * std + mean  # Denormalize
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(captions[i][:100] + '...' if len(captions[i]) > 100 else captions[i],
                         fontsize=10, wrap=True)

    plt.tight_layout()
    plt.savefig('output/sample_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: output/sample_visualization.png")
    plt.close()

def dataset_statistics(dataloader):
    """Show dataset statistics"""
    print("\nDataset Statistics:")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Number of batches: {len(dataloader)}")
    print(f"  Total samples: {len(dataloader.dataset)}")

    # Sample one batch to show structure
    batch = next(iter(dataloader))
    print(f"\nBatch structure:")
    print(f"  Images shape: {batch['images'].shape}")
    print(f"  Number of captions: {len(batch['captions'])}")
    print(f"\nSample caption:")
    print(f"  {batch['captions'][0]}")

def example_training_loop(dataloader, num_batches=3):
    """Example training loop"""
    print("\n" + "="*60)
    print("Example Training Loop")
    print("="*60)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        images = batch['images']
        captions = batch['captions']

        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images: {images.shape}")
        print(f"  Captions: {len(captions)}")

        # Your model training code would go here
        # loss = model(images, captions)
        # loss.backward()
        # optimizer.step()

    print("\n✓ Training loop example complete")

def main():
    print("="*60)
    print("  Parallel Synth Dataset Loader Example")
    print("="*60)

    # Check if dataset exists
    samples_dir = Path('./output/samples')
    if not samples_dir.exists() or not any(samples_dir.iterdir()):
        print("\n❌ No dataset found!")
        print("Generate samples first with:")
        print("  python examples/generate_first_batch.py")
        return 1

    print("\n1. Loading dataset...")

    # Create dataloader
    try:
        dataloader = create_dataloader(
            dataset_type='directory',
            data_path='./output/samples',
            batch_size=4,
            num_workers=2,
            image_size=512,
            augment=False,
            caption_type='medium',
            shuffle=True
        )
        print("✓ Dataset loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1

    # Show statistics
    print("\n2. Dataset statistics...")
    dataset_statistics(dataloader)

    # Visualize samples
    print("\n3. Visualizing samples...")
    try:
        visualize_batch(dataloader)
    except ImportError:
        print("⚠ Matplotlib not installed. Skipping visualization.")
        print("  Install with: pip install matplotlib")

    # Example training loop
    print("\n4. Example training loop...")
    example_training_loop(dataloader)

    print("\n" + "="*60)
    print("✓ Example complete!")
    print("="*60)

    print("\nNext steps:")
    print("  • Integrate with your model training code")
    print("  • Adjust batch size and image size as needed")
    print("  • Enable augmentation for training: augment=True")
    print("  • Try different caption types: 'short', 'long', 'technical', 'artistic'")
    print("")

    return 0

if __name__ == '__main__':
    sys.exit(main())
