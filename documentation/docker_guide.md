# Docker Deployment Guide

Run Parallel Synth dataset generation in Docker containers for easy deployment and scaling.

## Quick Start

### Build the Image

```bash
./scripts/docker_build.sh
```

### Run Generation

```bash
# Generate 10 test samples
docker run --rm --gpus all \
  -v $(pwd)/output:/app/output \
  parallelsynth/dataset-generator:latest

# Generate 100 samples with specific categories
docker run --rm --gpus all \
  -v $(pwd)/output:/app/output \
  parallelsynth/dataset-generator:latest \
  blender --background --python generators/blender_generator.py -- \
    --output /app/output/samples \
    --taxonomy /app/taxonomy/master_taxonomy.yaml \
    --count 100 \
    --categories geometry materials lighting camera
```

### Using Docker Compose

```bash
# Generate samples
docker-compose up generator

# Run validation
docker-compose --profile validate up validator

# Process samples
docker-compose --profile process up pipeline

# Upload to S3
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export S3_BUCKET=your-bucket
docker-compose --profile upload up uploader

# Start monitoring dashboard
docker-compose --profile monitor up -d monitor
# View at http://localhost:8080
```

## GPU Support

### NVIDIA GPU (Recommended)

Install NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Test GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## Distributed Rendering

Run multiple workers across GPUs:

```bash
# Start distributed workers
docker-compose --profile distributed up -d

# Scale workers
docker-compose --profile distributed up --scale worker-1=4 -d
```

## AWS Deployment

### EC2 with GPU

```bash
# Launch g5.xlarge instance with Deep Learning AMI
# SSH into instance

# Install Docker and NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y docker.io nvidia-container-toolkit

# Pull image
docker pull parallelsynth/dataset-generator:latest

# Run generation
docker run -d --gpus all \
  --name parallel-synth-worker \
  --restart unless-stopped \
  -v /data/output:/app/output \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  parallelsynth/dataset-generator:latest \
  python3 pipelines/distributed_render.py --count 10000
```

### ECS/Fargate

Create task definition:

```json
{
  "family": "parallel-synth-generator",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/parallel-synth-task-role",
  "containerDefinitions": [
    {
      "name": "generator",
      "image": "parallelsynth/dataset-generator:latest",
      "cpu": 4096,
      "memory": 16384,
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ],
      "command": [
        "python3",
        "pipelines/distributed_render.py",
        "--count",
        "1000"
      ],
      "environment": [
        {
          "name": "S3_BUCKET",
          "value": "parallel-synth-dataset"
        }
      ]
    }
  ]
}
```

### Kubernetes

Deploy to Kubernetes cluster:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: parallel-synth-generator
spec:
  replicas: 10
  selector:
    matchLabels:
      app: parallel-synth
  template:
    metadata:
      labels:
        app: parallel-synth
    spec:
      containers:
      - name: generator
        image: parallelsynth/dataset-generator:latest
        command:
        - python3
        - pipelines/distributed_render.py
        - --count
        - "1000"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        volumeMounts:
        - name: output
          mountPath: /app/output
        env:
        - name: S3_BUCKET
          value: "parallel-synth-dataset"
      volumes:
      - name: output
        persistentVolumeClaim:
          claimName: parallel-synth-pvc
```

## CI/CD Integration

### GitHub Actions

`.github/workflows/build-docker.yml`:

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          parallelsynth/dataset-generator:latest
          parallelsynth/dataset-generator:${{ github.sha }}
```

### GitLab CI

`.gitlab-ci.yml`:

```yaml
build:
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD
    - docker build -t parallelsynth/dataset-generator:latest .
    - docker push parallelsynth/dataset-generator:latest
```

## Custom Configuration

Mount custom config:

```bash
docker run --rm --gpus all \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/config/my_config.yaml:/app/config/production_config.yaml \
  parallelsynth/dataset-generator:latest \
  python3 pipelines/distributed_render.py \
    --config /app/config/production_config.yaml \
    --count 1000
```

## Monitoring

### Container Logs

```bash
# View logs
docker logs parallel-synth-generator -f

# Export logs
docker logs parallel-synth-generator > generation.log
```

### Resource Monitoring

```bash
# Real-time stats
docker stats

# Specific container
docker stats parallel-synth-generator
```

### Monitoring Dashboard

Run the monitoring service:

```bash
docker-compose --profile monitor up -d monitor
```

Access at: http://localhost:8080

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check container toolkit
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Restart Docker daemon
sudo systemctl restart docker
```

### Out of Memory

Increase Docker memory limit or reduce batch size:

```bash
# docker-compose.yml
services:
  generator:
    deploy:
      resources:
        limits:
          memory: 32G
```

### Slow Rendering

- Reduce sample count: `--count 64`
- Lower resolution in config
- Use CPU fallback if GPU busy

## Production Deployment

### Multi-Node Cluster

1. Set up Docker Swarm or Kubernetes
2. Deploy workers across nodes
3. Use shared storage (EFS, NFS)
4. Configure load balancing

### Cost Optimization

- Use Spot instances (70% discount)
- Auto-scaling based on queue
- S3 Intelligent Tiering
- Stop workers when idle

### Best Practices

1. **Version Control**: Tag images with version numbers
2. **Secrets Management**: Use Docker secrets or AWS Secrets Manager
3. **Health Checks**: Implement health check endpoints
4. **Logging**: Centralize logs (CloudWatch, ELK)
5. **Monitoring**: Use Prometheus/Grafana
6. **Backup**: Regular S3 syncs

## Registry Options

### Docker Hub

```bash
docker login
./scripts/docker_push.sh
```

### GitHub Container Registry

```bash
export DOCKER_USERNAME=ghcr.io/your-username
export REGISTRY=ghcr.io
docker login ghcr.io -u your-username
./scripts/docker_push.sh
```

### AWS ECR

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker tag parallelsynth/dataset-generator:latest \
  ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/parallel-synth:latest

docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/parallel-synth:latest
```

## Examples

### Example 1: Local Development

```bash
docker-compose up generator validator pipeline
```

### Example 2: AWS Production

```bash
# Build and push
./scripts/docker_build.sh
./scripts/docker_push.sh

# Deploy to ECS
aws ecs update-service \
  --cluster parallel-synth-cluster \
  --service generator-service \
  --force-new-deployment
```

### Example 3: Kubernetes Scale-Up

```bash
kubectl scale deployment parallel-synth-generator --replicas=50
kubectl get pods -w
```

---

**For more information, see the main documentation in `documentation/`**
