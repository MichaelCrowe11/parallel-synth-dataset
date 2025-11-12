# Scaling to 500 Million Samples

This guide outlines the infrastructure, strategy, and timeline for generating the complete Parallel Synth dataset of 500 million samples.

## Overview

**Target**: 500,000,000 samples
**Categories**: 11 major categories
**Estimated Timeline**: 6-12 months
**Estimated Cost**: $50,000 - $150,000 (AWS)

## Infrastructure Requirements

### Compute Resources

#### Option 1: AWS EC2 (Recommended)
```
Instance Types:
- g5.xlarge (NVIDIA A10G): $1.006/hour
- g5.2xlarge (NVIDIA A10G): $1.212/hour
- g5.4xlarge (NVIDIA A10G): $1.624/hour
- g5.8xlarge (NVIDIA A10G): $2.448/hour

Recommended Setup:
- 50x g5.2xlarge instances
- 24/7 operation
- Cost: ~$44,000/month
```

#### Option 2: On-Premises Render Farm
```
Hardware:
- 50-100 workstations with RTX 4090 GPUs
- High-speed networking (10Gbps+)
- NAS storage (500TB+)
- Backup power (UPS)

Initial Investment: $200,000 - $500,000
Ongoing Costs: Power, cooling, maintenance
```

#### Option 3: Hybrid (AWS + On-Prem)
```
- Use on-premises for baseline generation
- Burst to AWS for peak capacity
- Best cost/performance balance
```

### Storage Requirements

#### Sample Size Estimates
```
Per Sample:
- Image (PNG, 1920x1080): ~3-5 MB
- Metadata (JSON): ~5-10 KB
- Blend file (optional): ~2-10 MB
- Additional passes: ~10-20 MB

Average per sample: ~15 MB (with passes)
Minimal per sample: ~3-5 MB (image + metadata only)
```

#### Total Storage
```
Conservative (5 MB/sample):
500M × 5 MB = 2.5 PB

With render passes (15 MB/sample):
500M × 15 MB = 7.5 PB

Recommended: S3 Intelligent Tiering
- Frequent Access: ~10% of data
- Infrequent Access: ~90% of data
- Estimated monthly cost: $10,000 - $30,000
```

## Generation Strategy

### Phase 1: Foundation (Months 1-2)
**Target**: 10 million samples
**Focus**: Core categories and pipeline validation

```bash
Categories:
- Geometry: 2M samples
- Materials: 3M samples
- Lighting: 3M samples
- Camera: 2M samples

Daily Target: 170,000 samples
Instances: 10x g5.2xlarge
```

### Phase 2: Scale-Up (Months 3-6)
**Target**: 150 million samples
**Focus**: Full taxonomy coverage

```bash
Daily Target: 1.25M samples
Instances: 50x g5.2xlarge
Render Time per Sample: ~60 seconds
Throughput: 50 samples/second = 4.3M samples/day
```

### Phase 3: Full Production (Months 7-12)
**Target**: 340 million samples (remaining)
**Focus**: Balanced distribution across all categories

```bash
Daily Target: 1.9M samples
Instances: 75x g5.2xlarge
Complex Scenes: Add fluid and gas simulations
```

## Optimization Strategies

### 1. Adaptive Rendering
```python
# Simple scenes: Low samples
simple_scene.cycles.samples = 64

# Complex scenes: High samples
complex_scene.cycles.samples = 256

# Estimated time savings: 40%
```

### 2. Resolution Tiers
```python
# Standard resolution: 1920x1080 (75% of samples)
# High resolution: 3840x2160 (20% of samples)
# Ultra resolution: 7680x4320 (5% of samples)
```

### 3. Denoising
```python
# Use AI denoising to reduce sample count
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPTIX'  # GPU denoising

# Reduce samples from 256 to 128
# Time savings: 50%
```

### 4. Caching and Reuse
```python
# Cache common assets
# Reuse materials across scenes
# Procedural generation seeds for reproducibility

# Time savings: 30%
```

### 5. Priority Queue
```python
# High priority: Underrepresented categories
# Medium priority: Balanced generation
# Low priority: Overrepresented categories

# Ensures balanced dataset
```

## Cost Optimization

### AWS Cost Breakdown

#### Compute (g5.2xlarge, 50 instances, 6 months)
```
Hourly Rate: $1.212
Instances: 50
Hours: 4,320 per month
Monthly Cost: $262,080
6-Month Cost: $1,572,480

With Reserved Instances (1-year): 40% discount
6-Month Cost: ~$943,000
```

#### Storage (S3)
```
Data Size: 2.5 PB (conservative)

Standard Storage:
- First 50 TB: $0.023/GB = $1,150/month
- Next 450 TB: $0.022/GB = $9,900/month
- Over 500 TB: $0.021/GB = ~$42,000/month
Total: ~$53,000/month

With Intelligent Tiering:
- 10% Frequent Access: $5,300/month
- 90% Infrequent Access: $11,700/month (at $0.0125/GB)
Total: ~$17,000/month
6-Month Storage: ~$102,000
```

#### Data Transfer
```
Upload to S3: Free
Download (training): ~$90/TB
Estimated: $5,000/month
```

#### Total AWS Cost (6 months)
```
Compute: $943,000 (with reserved instances)
Storage: $102,000
Transfer: $30,000
TOTAL: ~$1,075,000

Per Sample Cost: $2.15
```

### Cost Reduction Strategies

1. **Spot Instances**: 70% discount
   - Compute cost: $283,000 (vs $943,000)
   - Total: $415,000 (vs $1,075,000)
   - **Per sample: $0.83**

2. **On-Premises**: Higher upfront, lower ongoing
   - Initial: $300,000 (hardware)
   - 6-Month Power: $30,000
   - Total: $330,000
   - **Per sample: $0.66**

3. **Hybrid Approach**: Best of both worlds
   - On-prem: 70% of generation
   - AWS Spot: 30% for bursts
   - Estimated: $350,000
   - **Per sample: $0.70**

## Timeline and Milestones

### Month 1: Infrastructure Setup
- [ ] Provision AWS resources or set up render farm
- [ ] Test generation pipeline at scale
- [ ] Validate quality control
- [ ] Set up monitoring and alerting
- **Milestone**: 5M samples generated

### Month 2: Pipeline Optimization
- [ ] Optimize render settings
- [ ] Implement caching
- [ ] Fine-tune quality thresholds
- **Milestone**: 15M samples total

### Months 3-4: Ramp Up
- [ ] Scale to 50 instances
- [ ] Generate 50M samples
- **Milestone**: 65M samples total

### Months 5-6: Full Production
- [ ] Scale to 75 instances
- [ ] Implement all categories
- [ ] Generate 100M samples
- **Milestone**: 165M samples total

### Months 7-9: Sustain and Balance
- [ ] Balance category distribution
- [ ] Quality improvements
- [ ] Generate 150M samples
- **Milestone**: 315M samples total

### Months 10-12: Final Push
- [ ] Generate remaining 185M samples
- [ ] Final validation
- [ ] Dataset packaging and release
- **Milestone**: 500M samples complete

## Monitoring and Quality Control

### Key Metrics

1. **Generation Rate**
   - Samples per hour
   - Samples per dollar
   - Failed generation rate

2. **Quality Metrics**
   - Average quality score
   - Validation pass rate
   - Caption quality

3. **Distribution Metrics**
   - Category balance
   - Subcategory coverage
   - Attribute diversity

### Monitoring Dashboard

```python
# Real-time dashboard showing:
# - Current generation rate
# - Total samples by category
# - Quality distribution
# - Cost per sample
# - Estimated completion date
# - Storage usage
```

### Automated Quality Control

```bash
# Run validation every 100K samples
parallel-synth-validate \
  --samples-dir ./output/batch_001 \
  --min-quality 0.7 \
  --report ./reports/batch_001.json

# Auto-retry failed generations
# Flag samples for manual review
# Maintain quality threshold >95% pass rate
```

## Distribution and Access

### Dataset Formats

1. **WebDataset** (Training)
   - Sharded tar files
   - Efficient streaming
   - Best for large-scale training

2. **Parquet** (Analysis)
   - Columnar format
   - Efficient querying
   - Best for data analysis

3. **JSONL** (Flexibility)
   - Line-delimited JSON
   - Easy to parse
   - Best for custom pipelines

### Access Methods

1. **S3 Direct Access**
   ```bash
   aws s3 cp s3://parallel-synth-dataset/samples/ . --recursive
   ```

2. **CloudFront CDN**
   - Low-latency access
   - Global distribution
   - Cache popular samples

3. **Academic Torrents**
   - Community distribution
   - Reduced bandwidth costs
   - Decentralized access

## Maintenance and Updates

### Continuous Improvement

- **Weekly**: Quality reports and metrics review
- **Monthly**: Category balance adjustments
- **Quarterly**: Pipeline optimizations and updates

### Version Control

```
v1.0: Initial 500M samples
v1.1: Quality improvements on low-scoring samples
v2.0: Additional categories and features
```

## Conclusion

Generating 500 million samples is an ambitious but achievable goal with:
- Proper infrastructure planning
- Cost optimization strategies
- Quality control processes
- Continuous monitoring

**Recommended Approach**: Hybrid on-premises + AWS Spot instances
**Estimated Total Cost**: $350,000 - $500,000
**Timeline**: 9-12 months
**Per-Sample Cost**: $0.70 - $1.00

This creates a world-class dataset for training next-generation AI models in 3D rendering and VFX!
