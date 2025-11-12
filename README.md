# Parallel Synth - 3D Rendering & VFX Dataset
## Target: 500 Million Samples

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Blender 3.6+](https://img.shields.io/badge/Blender-3.6+-orange.svg)](https://www.blender.org/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--5676--8816-green.svg)](https://orcid.org/0009-0008-5676-8816)

A comprehensive dataset scaffolding system for training multimodal AI models on 3D rendering, VFX, and computer graphics.

**Part of the Deep Parallel Workspace** - A unified AI model training infrastructure for Parallel Synth Media & Animation company.

**Created by:** [Michael Crowe](https://orcid.org/0009-0008-5676-8816) | **Organization:** Parallel Synth Media & Animation

## Dataset Overview

This dataset covers the complete spectrum of 3D computer graphics:
- **Camera** (50M samples): Angles, movements, lenses, focal lengths, depth of field
- **Lighting** (80M samples): Studio setups, HDRI, natural/artificial, color temperature
- **Materials** (100M samples): PBR, glass, metal, plastic, fabric, organic, subsurface scattering
- **Textures** (60M samples): Procedural, photorealistic, stylized, UV mapping
- **Liquids** (40M samples): Water, oil, paint, milk, viscosity variations
- **Gases** (35M samples): Smoke, fog, clouds, fire, volumetrics
- **Geometry** (45M samples): Primitives, complex meshes, topology, subdivision
- **Rendering** (40M samples): Ray tracing, path tracing, rasterization techniques
- **Post-Processing** (30M samples): Compositing, color grading, effects
- **Art Styles** (25M samples): Photorealism, stylized, NPR, cartoon, anime
- **Color Theory** (15M samples): Palettes, harmony, temperature, mood

## Data Formats

1. **Image-Text Pairs**: High-resolution renders with detailed technical captions
2. **Structured Metadata**: JSON/YAML with hierarchical parameters
3. **Code + Assets**: Blender scripts, USD files, procedural generation
4. **Video Sequences**: Frame sequences with temporal annotations

## Directory Structure

```
parallel_synth_dataset/
├── taxonomy/                    # Ontology and classification systems
├── schemas/                     # Data structure definitions
├── generators/                  # Procedural generation scripts
├── pipelines/                   # Data processing pipelines
├── samples/                     # Example outputs
├── aws_integration/             # S3 upload and management
├── quality_control/             # Validation and QC tools
└── documentation/               # Guides and references
```

## Use Cases

- Text-to-3D model training
- Image-to-3D model training
- Rendering parameter prediction
- Neural rendering engines
- VFX control models
- LLM fine-tuning on graphics knowledge

## Storage

All data stored in AWS S3 buckets with organized prefixes and metadata tagging.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate samples
blender --background --python generators/blender_generator.py -- \
  --output ./output/samples \
  --taxonomy ./taxonomy/master_taxonomy.yaml \
  --count 10

# 3. Validate quality
python quality_control/validator.py \
  --samples-dir ./output/samples \
  --report ./output/validation_report.json

# 4. Create training dataset
python pipelines/image_text_pipeline.py \
  --samples-dir ./output/samples \
  --output-dir ./output/training_data \
  --format all

# 5. Upload to S3
python aws_integration/s3_uploader.py \
  --bucket parallel-synth-dataset \
  --samples-dir ./output/samples \
  --category test_run
```

See `documentation/quickstart.md` for detailed instructions.
