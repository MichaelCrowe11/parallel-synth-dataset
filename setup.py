#!/usr/bin/env python3
"""Setup script for Parallel Synth Dataset"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
with open(requirements_file) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="parallel-synth-dataset",
    version="1.0.0",
    description="Comprehensive 3D rendering and VFX dataset for training multimodal AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Parallel Synth",
    author_email="contact@parallelsynth.com",
    url="https://github.com/MichaelCrowe11/parallel-synth-dataset",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="3d rendering vfx dataset ai machine-learning computer-graphics blender",
    entry_points={
        'console_scripts': [
            'parallel-synth-generate=generators.blender_generator:main',
            'parallel-synth-upload=aws_integration.s3_uploader:main',
            'parallel-synth-pipeline=pipelines.image_text_pipeline:main',
            'parallel-synth-validate=quality_control.validator:main',
            'parallel-synth-render=pipelines.distributed_render:main',
            'parallel-synth-extract=pipelines.video_frame_extractor:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
