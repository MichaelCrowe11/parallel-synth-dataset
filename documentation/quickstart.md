# Parallel Synth Dataset - Quick Start Guide

Welcome to the Parallel Synth 3D Rendering and VFX Dataset! This guide will get you up and running quickly.

## Overview

Parallel Synth is a comprehensive dataset generation system for training multimodal AI models on 3D rendering, VFX, and computer graphics. The dataset targets **500 million samples** across all aspects of 3D content creation.

## Prerequisites

### Software Requirements

1. **Python 3.8+**
2. **Blender 3.6+** (for procedural generation)
3. **CUDA-capable GPU** (recommended)
4. **AWS CLI** (for cloud storage)

### Installation

```bash
# Clone the repository
git clone https://github.com/MichaelCrowe11/parallel-synth-dataset.git
cd parallel-synth-dataset

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start: Generate Your First Samples

### 1. Generate Samples with Blender

```bash
# Generate 10 samples with basic categories
blender --background --python generators/blender_generator.py -- \
  --output ./output/samples \
  --taxonomy ./taxonomy/master_taxonomy.yaml \
  --count 10 \
  --categories geometry materials lighting camera
```

### 2. Validate Generated Samples

```bash
# Run quality control validation
python quality_control/validator.py \
  --samples-dir ./output/samples \
  --report ./output/validation_report.json \
  --min-quality 0.7
```

### 3. Create Training-Ready Dataset

```bash
# Convert to image-text pairs
python pipelines/image_text_pipeline.py \
  --samples-dir ./output/samples \
  --output-dir ./output/training_data \
  --format all \
  --split
```

### 4. Upload to AWS S3

```bash
# Upload to S3 bucket
python aws_integration/s3_uploader.py \
  --bucket parallel-synth-dataset \
  --samples-dir ./output/samples \
  --category geometry_materials \
  --create-bucket \
  --create-index
```

## Dataset Structure

```
parallel_synth_dataset/
├── taxonomy/               # Category definitions and ontology
├── schemas/               # Data structure schemas
├── generators/            # Blender generation scripts
├── pipelines/            # Data processing pipelines
├── aws_integration/      # S3 upload and management
├── quality_control/      # Validation tools
├── training/             # PyTorch dataloaders
└── documentation/        # Guides and references
```

## Key Components

### 1. Taxonomy

The taxonomy defines all categories and subcategories for the dataset:
- **Camera** (50M samples): angles, lenses, movements
- **Lighting** (80M samples): setups, HDRIs, atmospherics
- **Materials** (100M samples): PBR, glass, subsurface
- **Textures** (60M samples): procedural, photorealistic
- **Liquids** (40M samples): water, viscous fluids
- **Gases** (35M samples): smoke, fog, volumetrics
- **Geometry** (45M samples): meshes, topology
- **Rendering** (40M samples): techniques, engines
- **Post-processing** (30M samples): compositing, grading
- **Art Styles** (25M samples): realism, stylized, NPR
- **Color Theory** (15M samples): palettes, moods

See `taxonomy/master_taxonomy.yaml` for full details.

### 2. Metadata Schema

Each sample includes comprehensive metadata:
- Sample ID and timestamp
- Source and generation parameters
- File paths and formats
- Category assignments
- Multiple caption types (short, medium, long, technical, artistic)
- Quality metrics

See `schemas/metadata_schema.json` for full schema.

### 3. Generation Pipeline

1. **Procedural Generation**: Blender creates diverse 3D scenes
2. **Rendering**: Cycles/EEVEE render high-quality images
3. **Annotation**: Automatic caption and metadata generation
4. **Validation**: Quality control checks
5. **Export**: Multiple formats (WebDataset, JSONL, Parquet, CSV)
6. **Upload**: S3 storage with intelligent tiering

## Use Cases

### Training Vision Models

```python
from training.dataloader import create_dataloader

# Create dataloader for training
train_loader = create_dataloader(
    dataset_type='directory',
    data_path='./output/samples',
    batch_size=32,
    image_size=512,
    augment=True,
    caption_type='medium'
)

# Train your model
for batch in train_loader:
    images = batch['images']  # [B, 3, 512, 512]
    captions = batch['captions']  # List of strings
    # ... your training code
```

### Extracting Frames from CGI Films

```bash
# Extract frames from video
python pipelines/video_frame_extractor.py \
  --video path/to/cgi_film.mp4 \
  --output-dir ./output/frames \
  --fps 1 \
  --detect-shots \
  --annotate
```

### Distributed Rendering

```bash
# Render on multiple machines/GPUs
python pipelines/distributed_render.py \
  --config render_config.json \
  --count 1000 \
  --categories geometry materials lighting camera \
  --priority 5
```

## Configuration

### AWS Configuration

```bash
# Configure AWS credentials
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### Blender Configuration

Ensure Blender is in your PATH or specify the path:

```python
# In distributed_render.py config
{
  "workers": [
    {
      "hostname": "render-01",
      "gpu_ids": [0, 1],
      "blender_executable": "/path/to/blender",
      "max_concurrent_jobs": 2
    }
  ]
}
```

## Next Steps

1. **Scale Up**: Use distributed rendering to generate millions of samples
2. **Customize**: Modify taxonomy and generators for your specific needs
3. **Train Models**: Use the dataloaders to train your AI models
4. **Contribute**: Share improvements and additions

## Troubleshooting

### Blender Not Found

```bash
# Add Blender to PATH (macOS)
export PATH="/Applications/Blender.app/Contents/MacOS:$PATH"

# Or specify full path in commands
```

### Out of Memory

```bash
# Reduce batch size or image resolution
# In blender_generator.py, adjust:
self.scene.render.resolution_x = 512  # Lower resolution
self.scene.cycles.samples = 64  # Fewer samples
```

### S3 Upload Fails

```bash
# Check credentials
aws sts get-caller-identity

# Check bucket permissions
aws s3 ls s3://parallel-synth-dataset/
```

## Support

- Documentation: `documentation/`
- Issues: GitHub Issues
- Email: contact@parallelsynth.com

## License

This project is licensed under the MIT License.
