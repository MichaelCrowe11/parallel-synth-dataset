#!/usr/bin/env python3
"""
Example: Generate Your First Batch of Samples

This example shows you how to:
1. Generate a small batch of samples
2. Validate the quality
3. Create training-ready datasets
4. Upload to S3 (optional)
"""

import subprocess
import sys
from pathlib import Path
import json
import time

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg:^60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_step(step, msg):
    print(f"{Colors.BLUE}[{step}] {msg}{Colors.END}")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def check_blender():
    """Check if Blender is installed"""
    try:
        result = subprocess.run(['blender', '--version'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            return True, version
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check macOS location
    mac_blender = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if mac_blender.exists():
        return True, "Blender.app (macOS)"

    return False, None

def generate_samples(count=10):
    """Generate samples using Blender"""
    print_step("1/4", "Generating samples...")

    # Check Blender
    blender_found, blender_info = check_blender()
    if not blender_found:
        print_error("Blender not found!")
        print("Please install Blender 3.6+ from https://www.blender.org/")
        return False

    print_success(f"Found {blender_info}")

    # Determine Blender command
    blender_cmd = "blender"
    mac_blender = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if mac_blender.exists():
        blender_cmd = str(mac_blender)

    # Run generation
    cmd = [
        blender_cmd,
        '--background',
        '--python', 'generators/blender_generator.py',
        '--',
        '--output', './output/samples',
        '--taxonomy', './taxonomy/master_taxonomy.yaml',
        '--count', str(count),
        '--categories', 'geometry', 'materials', 'lighting', 'camera'
    ]

    print(f"\nGenerating {count} samples...")
    print("This may take a few minutes...\n")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time

        print_success(f"Generated {count} samples in {elapsed:.1f} seconds")
        print(f"Average: {elapsed/count:.1f}s per sample")
        return True

    except subprocess.CalledProcessError as e:
        print_error(f"Generation failed: {e}")
        return False

def validate_samples():
    """Validate generated samples"""
    print_step("2/4", "Validating samples...")

    cmd = [
        sys.executable,
        'quality_control/validator.py',
        '--samples-dir', './output/samples',
        '--report', './output/reports/validation_report.json',
        '--min-quality', '0.7'
    ]

    try:
        result = subprocess.run(cmd, check=True)

        # Read report
        report_path = Path('./output/reports/validation_report.json')
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)

            summary = report.get('summary', {})
            valid = summary.get('valid_samples', 0)
            total = summary.get('total_samples', 0)

            if total > 0:
                rate = (valid / total) * 100
                print_success(f"Validation: {valid}/{total} samples passed ({rate:.1f}%)")

        return True

    except subprocess.CalledProcessError:
        print_error("Validation failed")
        return False

def create_training_dataset():
    """Create training-ready dataset"""
    print_step("3/4", "Creating training dataset...")

    cmd = [
        sys.executable,
        'pipelines/image_text_pipeline.py',
        '--samples-dir', './output/samples',
        '--output-dir', './output/training_data',
        '--format', 'all',
        '--split'
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print_success("Training dataset created")
        return True
    except subprocess.CalledProcessError:
        print_error("Dataset creation failed")
        return False

def show_results():
    """Show summary of results"""
    print_step("4/4", "Summary")

    samples_dir = Path('./output/samples')
    training_dir = Path('./output/training_data')

    if samples_dir.exists():
        sample_count = len(list(samples_dir.iterdir()))
        print(f"\n📁 Samples directory: {samples_dir}")
        print(f"   {sample_count} samples generated")

    if training_dir.exists():
        print(f"\n📊 Training data: {training_dir}")

        # Check for different formats
        formats = []
        if (training_dir / "parallel_synth_medium.jsonl").exists():
            formats.append("JSONL")
        if (training_dir / "parallel_synth_dataset.parquet").exists():
            formats.append("Parquet")
        if list(training_dir.glob("*.tar")):
            formats.append("WebDataset")

        if formats:
            print(f"   Formats: {', '.join(formats)}")

    report_path = Path('./output/reports/validation_report.json')
    if report_path.exists():
        print(f"\n📋 Validation report: {report_path}")

    print("\n" + "="*60)
    print_success("First batch complete! 🎉")
    print("="*60 + "\n")

    print("Next steps:")
    print("  • View samples: open output/samples/")
    print("  • Load training data: see examples/load_dataset.py")
    print("  • Scale up: see documentation/scaling_to_500m.md")
    print("  • Upload to S3: python aws_integration/s3_uploader.py --help")
    print("")

def main():
    print_header("Parallel Synth - First Batch Generator")

    # Get count from user
    try:
        count = int(input("How many samples to generate? [10]: ") or "10")
    except ValueError:
        count = 10

    print(f"\nGenerating {count} samples...\n")

    # Run pipeline
    if not generate_samples(count):
        print_error("Generation failed. Check Blender installation.")
        return 1

    if not validate_samples():
        print_warning("Validation had issues, but continuing...")

    if not create_training_dataset():
        print_warning("Dataset creation had issues, but continuing...")

    show_results()

    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
