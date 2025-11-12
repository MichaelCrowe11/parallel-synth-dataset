#!/usr/bin/env python3
"""
Parallel Synth - AWS S3 Upload Manager
Efficiently uploads generated samples to S3 with metadata tagging
"""

import boto3
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
from botocore.exceptions import ClientError
from tqdm import tqdm


class S3DatasetUploader:
    """Manages dataset uploads to AWS S3"""

    def __init__(self, bucket_name: str, region: str = 'us-east-1', profile: str = None):
        """
        Initialize S3 uploader

        Args:
            bucket_name: S3 bucket name
            region: AWS region
            profile: AWS profile name (optional)
        """
        self.bucket_name = bucket_name
        self.region = region

        # Initialize boto3 session
        if profile:
            session = boto3.Session(profile_name=profile)
            self.s3_client = session.client('s3', region_name=region)
        else:
            self.s3_client = boto3.client('s3', region_name=region)

        self.uploaded_files = []
        self.failed_uploads = []

    def create_bucket_if_not_exists(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✓ Bucket '{self.bucket_name}' already exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print(f"Creating bucket '{self.bucket_name}'...")
                if self.region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
                    )
                print(f"✓ Bucket '{self.bucket_name}' created")
            else:
                raise

    def setup_bucket_lifecycle(self):
        """Configure S3 bucket lifecycle policies for cost optimization"""
        lifecycle_config = {
            'Rules': [
                {
                    'Id': 'TransitionToIA',
                    'Status': 'Enabled',
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 90,
                            'StorageClass': 'INTELLIGENT_TIERING'
                        }
                    ],
                    'Filter': {'Prefix': 'samples/'}
                }
            ]
        }

        try:
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket_name,
                LifecycleConfiguration=lifecycle_config
            )
            print("✓ Lifecycle policy configured")
        except ClientError as e:
            print(f"Warning: Could not set lifecycle policy: {e}")

    def upload_sample(self, sample_dir: Path, category: str, compression: bool = True) -> Dict:
        """
        Upload a single sample with all its files

        Args:
            sample_dir: Directory containing sample files
            category: Category name for S3 organization
            compression: Whether to compress files

        Returns:
            Upload metadata
        """
        sample_id = sample_dir.name
        s3_prefix = f"samples/{category}/{sample_id}"

        # Read metadata
        metadata_file = sample_dir / f"{sample_id}.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        uploaded_files = {}

        # Upload all files in the sample directory
        for file_path in sample_dir.glob('*'):
            if file_path.is_file():
                s3_key = f"{s3_prefix}/{file_path.name}"

                # Determine content type
                content_type = self.get_content_type(file_path.suffix)

                # Calculate MD5 for verification
                md5_hash = self.calculate_md5(file_path)

                # Upload file
                extra_args = {
                    'ContentType': content_type,
                    'Metadata': {
                        'sample-id': sample_id,
                        'category': category,
                        'md5': md5_hash,
                        'upload-timestamp': datetime.utcnow().isoformat()
                    }
                }

                # Add tags
                tags = self.build_tags(metadata)
                if tags:
                    extra_args['Tagging'] = tags

                try:
                    self.s3_client.upload_file(
                        str(file_path),
                        self.bucket_name,
                        s3_key,
                        ExtraArgs=extra_args
                    )

                    uploaded_files[file_path.suffix] = {
                        's3_key': s3_key,
                        's3_uri': f"s3://{self.bucket_name}/{s3_key}",
                        'size': file_path.stat().st_size,
                        'md5': md5_hash
                    }

                except ClientError as e:
                    print(f"✗ Failed to upload {file_path.name}: {e}")
                    raise

        return {
            'sample_id': sample_id,
            'category': category,
            's3_prefix': s3_prefix,
            'files': uploaded_files,
            'upload_timestamp': datetime.utcnow().isoformat()
        }

    def upload_batch(self, samples_dir: Path, category: str, max_workers: int = 10) -> Dict:
        """
        Upload multiple samples in parallel

        Args:
            samples_dir: Directory containing sample subdirectories
            category: Category name
            max_workers: Number of parallel upload threads

        Returns:
            Upload summary
        """
        sample_dirs = [d for d in samples_dir.iterdir() if d.is_dir()]

        print(f"\nUploading {len(sample_dirs)} samples from {samples_dir}...")
        print(f"Category: {category}")
        print(f"Bucket: s3://{self.bucket_name}/samples/{category}/")
        print(f"Workers: {max_workers}\n")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.upload_sample, sample_dir, category): sample_dir
                for sample_dir in sample_dirs
            }

            with tqdm(total=len(sample_dirs), desc="Uploading") as pbar:
                for future in as_completed(futures):
                    sample_dir = futures[future]
                    try:
                        result = future.result()
                        self.uploaded_files.append(result)
                        pbar.update(1)
                    except Exception as e:
                        self.failed_uploads.append({
                            'sample_dir': str(sample_dir),
                            'error': str(e)
                        })
                        pbar.update(1)

        summary = {
            'total_samples': len(sample_dirs),
            'successful': len(self.uploaded_files),
            'failed': len(self.failed_uploads),
            'bucket': self.bucket_name,
            'category': category,
            'timestamp': datetime.utcnow().isoformat()
        }

        print(f"\n✓ Upload complete!")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")

        if self.failed_uploads:
            print("\nFailed uploads:")
            for failure in self.failed_uploads:
                print(f"  - {failure['sample_dir']}: {failure['error']}")

        return summary

    def build_tags(self, metadata: Dict) -> str:
        """Build S3 tags from metadata"""
        tags = []

        # Add category tags
        if 'categories' in metadata:
            categories = metadata['categories']

            if 'camera' in categories:
                tags.append(f"camera-angle={categories['camera'].get('angle', 'unknown')}")

            if 'lighting' in categories:
                tags.append(f"lighting={categories['lighting'].get('setup_type', 'unknown')}")

            if 'materials' in categories:
                tags.append(f"material={categories['materials'].get('primary_material', 'unknown')}")

            if 'geometry' in categories:
                tags.append(f"geometry={categories['geometry'].get('type', 'unknown')}")

        # Limit to 10 tags (AWS limit)
        tags = tags[:10]

        return '&'.join(tags) if tags else ''

    @staticmethod
    def get_content_type(suffix: str) -> str:
        """Get content type for file extension"""
        content_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.exr': 'image/x-exr',
            '.tiff': 'image/tiff',
            '.json': 'application/json',
            '.blend': 'application/octet-stream',
            '.usd': 'application/octet-stream',
            '.usda': 'text/plain',
            '.fbx': 'application/octet-stream',
            '.gltf': 'model/gltf+json',
            '.glb': 'model/gltf-binary',
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime'
        }
        return content_types.get(suffix.lower(), 'application/octet-stream')

    @staticmethod
    def calculate_md5(file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def save_upload_manifest(self, output_path: Path):
        """Save upload manifest to file"""
        manifest = {
            'uploaded_files': self.uploaded_files,
            'failed_uploads': self.failed_uploads,
            'summary': {
                'total': len(self.uploaded_files) + len(self.failed_uploads),
                'successful': len(self.uploaded_files),
                'failed': len(self.failed_uploads)
            },
            'timestamp': datetime.utcnow().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\n✓ Upload manifest saved to {output_path}")

    def create_dataset_index(self, category: str) -> Dict:
        """Create an index of all samples in a category"""
        prefix = f"samples/{category}/"

        print(f"Creating index for category: {category}")

        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

        samples = {}

        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                # Extract sample ID from key
                parts = key.split('/')
                if len(parts) >= 3:
                    sample_id = parts[2]

                    if sample_id not in samples:
                        samples[sample_id] = {
                            'sample_id': sample_id,
                            'files': []
                        }

                    samples[sample_id]['files'].append({
                        's3_key': key,
                        's3_uri': f"s3://{self.bucket_name}/{key}",
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat()
                    })

        index = {
            'category': category,
            'sample_count': len(samples),
            'samples': list(samples.values()),
            'generated_at': datetime.utcnow().isoformat()
        }

        # Upload index to S3
        index_key = f"indexes/{category}_index.json"
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=index_key,
            Body=json.dumps(index, indent=2),
            ContentType='application/json'
        )

        print(f"✓ Index created: s3://{self.bucket_name}/{index_key}")
        print(f"  Samples indexed: {len(samples)}")

        return index


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Parallel Synth S3 Uploader')
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket name')
    parser.add_argument('--samples-dir', type=str, required=True, help='Directory containing samples')
    parser.add_argument('--category', type=str, required=True, help='Category name')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')
    parser.add_argument('--profile', type=str, help='AWS profile name')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers')
    parser.add_argument('--create-bucket', action='store_true', help='Create bucket if it does not exist')
    parser.add_argument('--create-index', action='store_true', help='Create dataset index after upload')
    parser.add_argument('--manifest', type=str, help='Path to save upload manifest')

    args = parser.parse_args()

    uploader = S3DatasetUploader(args.bucket, args.region, args.profile)

    if args.create_bucket:
        uploader.create_bucket_if_not_exists()
        uploader.setup_bucket_lifecycle()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        return 1

    summary = uploader.upload_batch(samples_dir, args.category, args.workers)

    if args.manifest:
        uploader.save_upload_manifest(Path(args.manifest))

    if args.create_index:
        uploader.create_dataset_index(args.category)

    return 0 if summary['failed'] == 0 else 1


if __name__ == '__main__':
    exit(main())
