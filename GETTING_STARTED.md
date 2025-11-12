# Getting Started with Parallel Synth Dataset Generation

Welcome! This guide will help you generate your first 3D rendering samples in minutes.

## Prerequisites

Before starting, ensure you have:

1. **Python 3.8+** installed
2. **Blender 3.6+** installed ([Download](https://www.blender.org/download/))
3. **Git** (already installed if you cloned this repo)
4. **8GB+ RAM** (16GB+ recommended)
5. **GPU with 4GB+ VRAM** (optional but recommended)

## Quick Start (3 Easy Steps)

### Option 1: Interactive Start Script

```bash
./scripts/start.sh
```

This interactive script will:
- Check your installation
- Guide you through generating samples
- Validate quality
- Create training datasets

### Option 2: Python Quick Start

```bash
python3 examples/generate_first_batch.py
```

Generates 10 test samples and processes them automatically.

### Option 3: Manual Commands

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Generate 10 samples
blender --background --python generators/blender_generator.py -- \
  --output ./output/samples \
  --taxonomy ./taxonomy/master_taxonomy.yaml \
  --count 10 \
  --categories geometry materials lighting camera

# 3. Validate
python3 quality_control/validator.py \
  --samples-dir ./output/samples \
  --report ./output/reports/validation.json

# 4. Create training dataset
python3 pipelines/image_text_pipeline.py \
  --samples-dir ./output/samples \
  --output-dir ./output/training_data \
  --format all --split
```

## Your First Sample

After running the quick start, you'll have:

```
output/
├── samples/                    # Generated 3D renders
│   └── [uuid]/
│       ├── [uuid].png         # Rendered image
│       ├── [uuid].json        # Metadata
│       └── [uuid].blend       # Blender file
├── training_data/             # Training-ready datasets
│   ├── *.tar                  # WebDataset format
│   ├── *.jsonl                # JSONL format
│   └── *.parquet              # Parquet format
└── reports/                   # Quality reports
    └── validation_report.json
```

## Load and Use Your Dataset

```python
from training.dataloader import create_dataloader

# Create PyTorch dataloader
train_loader = create_dataloader(
    dataset_type='directory',
    data_path='./output/samples',
    batch_size=16,
    image_size=512,
    augment=True,
    caption_type='medium'
)

# Use in training
for batch in train_loader:
    images = batch['images']  # [B, 3, 512, 512]
    captions = batch['captions']
    # Your training code here...
```

See `examples/load_dataset.py` for a complete example.

## Monitor Progress

Watch real-time generation statistics:

```bash
python3 scripts/monitor.py --samples-dir ./output/samples --refresh 5
```

Shows:
- Generation rate (samples/hour)
- Quality metrics
- Cost estimates
- Estimated completion time
- Category distribution

## Troubleshooting

### Blender Not Found

**macOS:**
```bash
export PATH="/Applications/Blender.app/Contents/MacOS:$PATH"
```

**Linux:**
```bash
sudo apt install blender
# or download from blender.org
```

**Windows:**
Add Blender to your PATH or use full path in commands.

### Out of Memory

Reduce resolution in `generators/blender_generator.py`:

```python
self.scene.render.resolution_x = 1280  # Lower from 1920
self.scene.render.resolution_y = 720   # Lower from 1080
self.scene.cycles.samples = 64          # Lower from 128
```

### Dependencies Missing

```bash
pip3 install -r requirements.txt
```

### Slow Generation

- Enable GPU rendering (automatic if available)
- Reduce sample count: `--count 64` instead of 128
- Enable denoising (already enabled by default)
- Use lower resolution for testing

## Next Steps

### 1. Generate More Samples

```bash
# Generate 100 samples
python3 examples/generate_first_batch.py
# Enter: 100

# Or use the start script
./scripts/start.sh
# Choose option 2
```

### 2. Try Different Categories

```bash
blender --background --python generators/blender_generator.py -- \
  --output ./output/samples \
  --taxonomy ./taxonomy/master_taxonomy.yaml \
  --count 50 \
  --categories liquids gases  # Try fluids and volumetrics!
```

Available categories:
- `geometry` - 3D shapes and meshes
- `materials` - PBR materials (metal, glass, etc.)
- `lighting` - Light setups and HDRIs
- `camera` - Angles and focal lengths
- `textures` - Procedural and realistic textures
- `liquids` - Water and fluid simulations
- `gases` - Smoke, fog, and volumetrics
- `rendering` - Different render techniques
- `post_processing` - Effects and grading
- `art_styles` - Artistic styles
- `color` - Color palettes and theory

### 3. Set Up Distributed Rendering

Edit `config/production_config.yaml` to add render nodes:

```yaml
distributed:
  workers:
    - hostname: "render-node-01"
      gpu_ids: [0, 1]
      blender_executable: "/usr/local/bin/blender"
      max_concurrent_jobs: 2
```

Then run:

```bash
python3 pipelines/distributed_render.py \
  --config config/production_config.yaml \
  --count 1000
```

### 4. Upload to AWS S3

```bash
# Configure AWS
aws configure

# Upload samples
python3 aws_integration/s3_uploader.py \
  --bucket my-parallel-synth-dataset \
  --samples-dir ./output/samples \
  --category test_batch \
  --create-bucket \
  --create-index
```

### 5. Scale to 500M

See `documentation/scaling_to_500m.md` for the full production plan:
- Infrastructure setup
- Cost optimization
- Timeline and milestones
- Distributed rendering at scale

## Examples

### Example 1: Load Dataset

```bash
python3 examples/load_dataset.py
```

### Example 2: Generate Specific Materials

```bash
blender --background --python generators/blender_generator.py -- \
  --output ./output/glass_samples \
  --taxonomy ./taxonomy/master_taxonomy.yaml \
  --count 20 \
  --categories materials
```

### Example 3: Process Existing Samples

```bash
# Validate
python3 quality_control/validator.py \
  --samples-dir ./output/samples \
  --min-quality 0.8

# Create training data
python3 pipelines/image_text_pipeline.py \
  --samples-dir ./output/samples \
  --output-dir ./output/training_data \
  --format webdataset
```

## Resources

- **Quick Start**: `documentation/quickstart.md`
- **Scaling Guide**: `documentation/scaling_to_500m.md`
- **Deep Parallel Workspace**: `DEEP_PARALLEL_WORKSPACE.md`
- **Taxonomy**: `taxonomy/master_taxonomy.yaml`
- **Config**: `config/production_config.yaml`

## Get Help

- Check `documentation/` for detailed guides
- Review `examples/` for working code
- Open an issue on GitHub
- Email: contact@parallelsynth.com

## Ready to Scale?

Once you've tested with small batches, see `documentation/scaling_to_500m.md` for:
- Infrastructure planning ($350K-$500K total cost)
- Distributed rendering setup
- AWS optimization strategies
- 9-12 month timeline to 500M samples

---

**Let's build the future of AI-assisted content creation! 🎬🤖✨**
