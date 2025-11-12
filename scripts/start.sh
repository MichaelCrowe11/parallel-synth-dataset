#!/bin/bash
# Parallel Synth - Quick Start Script
# This script helps you get started with dataset generation

set -e

echo "=================================================="
echo "  Parallel Synth Dataset - Quick Start"
echo "  Target: 500 Million Samples"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python
echo -e "${BLUE}[1/6] Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check Blender
echo -e "\n${BLUE}[2/6] Checking Blender installation...${NC}"
if command -v blender &> /dev/null; then
    BLENDER_VERSION=$(blender --version | head -n 1)
    echo -e "${GREEN}✓ Found: $BLENDER_VERSION${NC}"
    BLENDER_CMD="blender"
elif [ -f "/Applications/Blender.app/Contents/MacOS/Blender" ]; then
    echo -e "${GREEN}✓ Found: Blender.app (macOS)${NC}"
    BLENDER_CMD="/Applications/Blender.app/Contents/MacOS/Blender"
    export PATH="/Applications/Blender.app/Contents/MacOS:$PATH"
else
    echo -e "${YELLOW}⚠ Blender not found in PATH${NC}"
    echo "  Please install Blender 3.6+ from https://www.blender.org/"
    echo "  Or specify path with: export BLENDER_PATH=/path/to/blender"
    BLENDER_CMD="${BLENDER_PATH:-blender}"
fi

# Check dependencies
echo -e "\n${BLUE}[3/6] Checking Python dependencies...${NC}"
if python3 -c "import numpy, PIL, yaml" 2>/dev/null; then
    echo -e "${GREEN}✓ Core dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Installing dependencies...${NC}"
    pip install -r requirements.txt
fi

# Create directories
echo -e "\n${BLUE}[4/6] Setting up directories...${NC}"
mkdir -p output/{samples,training_data,reports,logs,cache}
mkdir -p examples
echo -e "${GREEN}✓ Directories created${NC}"

# Check AWS credentials (optional)
echo -e "\n${BLUE}[5/6] Checking AWS credentials (optional)...${NC}"
if command -v aws &> /dev/null; then
    if aws sts get-caller-identity &> /dev/null 2>&1; then
        echo -e "${GREEN}✓ AWS credentials configured${NC}"
        AWS_READY=true
    else
        echo -e "${YELLOW}⚠ AWS credentials not configured${NC}"
        echo "  Run 'aws configure' to set up S3 uploads"
        AWS_READY=false
    fi
else
    echo -e "${YELLOW}⚠ AWS CLI not installed${NC}"
    echo "  Install from: https://aws.amazon.com/cli/"
    AWS_READY=false
fi

echo -e "\n${BLUE}[6/6] Setup complete!${NC}"
echo ""
echo "=================================================="
echo -e "${GREEN}  Ready to Generate Samples!${NC}"
echo "=================================================="
echo ""

# Ask user what they want to do
echo "What would you like to do?"
echo ""
echo "  1) Generate 10 test samples (Quick test)"
echo "  2) Generate 100 samples (Small batch)"
echo "  3) Generate 1000 samples (Large batch)"
echo "  4) Start distributed rendering"
echo "  5) Process existing samples"
echo "  6) Exit"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}Generating 10 test samples...${NC}\n"
        $BLENDER_CMD --background --python generators/blender_generator.py -- \
            --output ./output/samples \
            --taxonomy ./taxonomy/master_taxonomy.yaml \
            --count 10 \
            --categories geometry materials lighting camera

        echo -e "\n${GREEN}✓ Generation complete!${NC}"
        echo ""
        echo "Next steps:"
        echo "  - Validate: python quality_control/validator.py --samples-dir ./output/samples --report ./output/reports/validation.json"
        echo "  - Process: python pipelines/image_text_pipeline.py --samples-dir ./output/samples --output-dir ./output/training_data --format all"
        ;;

    2)
        echo -e "\n${GREEN}Generating 100 samples...${NC}\n"
        $BLENDER_CMD --background --python generators/blender_generator.py -- \
            --output ./output/samples \
            --taxonomy ./taxonomy/master_taxonomy.yaml \
            --count 100 \
            --categories geometry materials lighting camera textures

        echo -e "\n${GREEN}✓ Generation complete!${NC}"
        ;;

    3)
        echo -e "\n${GREEN}Generating 1000 samples...${NC}\n"
        echo "This will take approximately 15-30 minutes..."
        $BLENDER_CMD --background --python generators/blender_generator.py -- \
            --output ./output/samples \
            --taxonomy ./taxonomy/master_taxonomy.yaml \
            --count 1000 \
            --categories geometry materials lighting camera textures

        echo -e "\n${GREEN}✓ Generation complete!${NC}"
        ;;

    4)
        echo -e "\n${GREEN}Starting distributed rendering...${NC}\n"
        if [ -f "config/render_config.json" ]; then
            python pipelines/distributed_render.py --config config/render_config.json --count 1000
        else
            echo "Creating default config..."
            python pipelines/distributed_render.py --count 100
        fi
        ;;

    5)
        echo -e "\n${GREEN}Processing existing samples...${NC}\n"

        # Count samples
        SAMPLE_COUNT=$(find ./output/samples -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
        echo "Found $SAMPLE_COUNT samples"

        if [ "$SAMPLE_COUNT" -gt 0 ]; then
            # Validate
            echo -e "\n${BLUE}Running validation...${NC}"
            python quality_control/validator.py \
                --samples-dir ./output/samples \
                --report ./output/reports/validation_report.json \
                --min-quality 0.7

            # Create training dataset
            echo -e "\n${BLUE}Creating training dataset...${NC}"
            python pipelines/image_text_pipeline.py \
                --samples-dir ./output/samples \
                --output-dir ./output/training_data \
                --format all \
                --split

            echo -e "\n${GREEN}✓ Processing complete!${NC}"

            # Ask about S3 upload
            if [ "$AWS_READY" = true ]; then
                read -p "Upload to S3? (y/n): " upload
                if [ "$upload" = "y" ]; then
                    read -p "Enter S3 bucket name: " bucket
                    python aws_integration/s3_uploader.py \
                        --bucket "$bucket" \
                        --samples-dir ./output/samples \
                        --category test_batch \
                        --create-bucket
                fi
            fi
        else
            echo -e "${YELLOW}No samples found. Generate some first!${NC}"
        fi
        ;;

    6)
        echo "Exiting..."
        exit 0
        ;;

    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo -e "${GREEN}  Complete!${NC}"
echo "=================================================="
echo ""
echo "View your samples in: ./output/samples/"
echo "Training data in: ./output/training_data/"
echo "Reports in: ./output/reports/"
echo ""
echo "For more options, see: documentation/quickstart.md"
echo ""
