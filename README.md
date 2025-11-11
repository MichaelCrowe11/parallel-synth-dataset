# Parallel Synth - 3D Rendering & VFX Dataset
## Target: 500 Million Samples

A comprehensive dataset scaffolding system for training multimodal AI models on 3D rendering, VFX, and computer graphics.

## Dataset Overview

This dataset covers the complete spectrum of 3D computer graphics:
- **Camera**: Angles, movements, lenses, focal lengths, depth of field
- **Lighting**: Studio setups, HDRI, natural/artificial, color temperature
- **Materials**: PBR, glass, metal, plastic, fabric, organic, subsurface scattering
- **Textures**: Procedural, photorealistic, stylized, UV mapping
- **Liquids**: Water, oil, paint, milk, viscosity variations
- **Gases**: Smoke, fog, clouds, fire, volumetrics
- **Geometry**: Primitives, complex meshes, topology, subdivision
- **Rendering**: Ray tracing, path tracing, rasterization techniques
- **Post-Processing**: Compositing, color grading, effects
- **Art Styles**: Photorealism, stylized, NPR, cartoon, anime
- **Color Theory**: Palettes, harmony, temperature, mood

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

## Getting Started

See `documentation/quickstart.md` for setup instructions.
