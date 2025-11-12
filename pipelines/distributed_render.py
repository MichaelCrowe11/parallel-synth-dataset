#!/usr/bin/env python3
"""
Parallel Synth - Distributed Rendering Orchestration
Coordinates Blender rendering across multiple machines/GPUs
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
import threading
import queue
from enum import Enum
import socket


class RenderStatus(Enum):
    """Render job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RenderJob:
    """Represents a single render job"""
    job_id: str
    script_path: str
    output_dir: str
    categories: List[str]
    seed: Optional[int]
    priority: int
    status: RenderStatus
    worker_id: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    render_time: Optional[float]


@dataclass
class RenderWorker:
    """Represents a render worker (machine/GPU)"""
    worker_id: str
    hostname: str
    gpu_id: Optional[int]
    blender_executable: str
    max_concurrent_jobs: int
    status: str
    current_job: Optional[str]
    jobs_completed: int
    jobs_failed: int
    total_render_time: float


class DistributedRenderOrchestrator:
    """Orchestrates distributed rendering across multiple workers"""

    def __init__(self, config_path: Optional[Path] = None):
        self.job_queue = queue.PriorityQueue()
        self.jobs: Dict[str, RenderJob] = {}
        self.workers: Dict[str, RenderWorker] = {}
        self.running = False

        # Load configuration
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.default_config()

        # Initialize workers
        self.initialize_workers()

    def default_config(self) -> Dict:
        """Default configuration"""
        return {
            'workers': [
                {
                    'hostname': socket.gethostname(),
                    'gpu_ids': [0],
                    'blender_executable': 'blender',
                    'max_concurrent_jobs': 1
                }
            ],
            'output_dir': './output',
            'blender_script': './generators/blender_generator.py',
            'taxonomy': './taxonomy/master_taxonomy.yaml'
        }

    def initialize_workers(self):
        """Initialize render workers from configuration"""
        print("Initializing render workers...")

        for worker_config in self.config['workers']:
            hostname = worker_config['hostname']
            gpu_ids = worker_config.get('gpu_ids', [None])
            blender_exe = worker_config.get('blender_executable', 'blender')
            max_concurrent = worker_config.get('max_concurrent_jobs', 1)

            for gpu_id in gpu_ids:
                worker_id = f"{hostname}_gpu{gpu_id}" if gpu_id is not None else f"{hostname}_cpu"

                worker = RenderWorker(
                    worker_id=worker_id,
                    hostname=hostname,
                    gpu_id=gpu_id,
                    blender_executable=blender_exe,
                    max_concurrent_jobs=max_concurrent,
                    status='idle',
                    current_job=None,
                    jobs_completed=0,
                    jobs_failed=0,
                    total_render_time=0.0
                )

                self.workers[worker_id] = worker
                print(f"  ✓ Worker registered: {worker_id}")

        print(f"✓ {len(self.workers)} workers initialized")

    def create_job(
        self,
        categories: List[str],
        seed: Optional[int] = None,
        priority: int = 5
    ) -> str:
        """
        Create a new render job

        Args:
            categories: List of taxonomy categories to include
            seed: Random seed for reproducibility
            priority: Job priority (lower = higher priority)

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())

        job = RenderJob(
            job_id=job_id,
            script_path=self.config['blender_script'],
            output_dir=self.config['output_dir'],
            categories=categories,
            seed=seed,
            priority=priority,
            status=RenderStatus.PENDING,
            worker_id=None,
            created_at=datetime.utcnow().isoformat(),
            started_at=None,
            completed_at=None,
            error_message=None,
            render_time=None
        )

        self.jobs[job_id] = job
        self.job_queue.put((priority, time.time(), job_id))

        return job_id

    def create_batch_jobs(
        self,
        categories: List[str],
        count: int,
        priority: int = 5
    ) -> List[str]:
        """Create multiple render jobs"""
        print(f"\nCreating {count} render jobs...")

        job_ids = []
        for i in range(count):
            job_id = self.create_job(categories, seed=None, priority=priority)
            job_ids.append(job_id)

        print(f"✓ Created {len(job_ids)} jobs")
        return job_ids

    def get_available_worker(self) -> Optional[RenderWorker]:
        """Get an available worker"""
        for worker in self.workers.values():
            if worker.status == 'idle':
                return worker
        return None

    def execute_job(self, job: RenderJob, worker: RenderWorker):
        """Execute a render job on a worker"""
        print(f"\n[{worker.worker_id}] Starting job {job.job_id}")

        # Update status
        job.status = RenderStatus.RUNNING
        job.worker_id = worker.worker_id
        job.started_at = datetime.utcnow().isoformat()

        worker.status = 'busy'
        worker.current_job = job.job_id

        # Build Blender command
        cmd = self.build_blender_command(job, worker)

        print(f"[{worker.worker_id}] Command: {' '.join(cmd)}")

        start_time = time.time()

        try:
            # Execute Blender
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            render_time = time.time() - start_time

            if result.returncode == 0:
                # Success
                job.status = RenderStatus.COMPLETED
                job.completed_at = datetime.utcnow().isoformat()
                job.render_time = render_time

                worker.jobs_completed += 1
                worker.total_render_time += render_time

                print(f"[{worker.worker_id}] ✓ Job {job.job_id} completed in {render_time:.2f}s")

            else:
                # Failed
                job.status = RenderStatus.FAILED
                job.error_message = result.stderr
                worker.jobs_failed += 1

                print(f"[{worker.worker_id}] ✗ Job {job.job_id} failed")
                print(f"Error: {result.stderr}")

        except subprocess.TimeoutExpired:
            job.status = RenderStatus.FAILED
            job.error_message = "Render timeout exceeded"
            worker.jobs_failed += 1
            print(f"[{worker.worker_id}] ✗ Job {job.job_id} timed out")

        except Exception as e:
            job.status = RenderStatus.FAILED
            job.error_message = str(e)
            worker.jobs_failed += 1
            print(f"[{worker.worker_id}] ✗ Job {job.job_id} error: {e}")

        finally:
            # Reset worker status
            worker.status = 'idle'
            worker.current_job = None

    def build_blender_command(self, job: RenderJob, worker: RenderWorker) -> List[str]:
        """Build Blender command for job"""
        cmd = [
            worker.blender_executable,
            '--background',
            '--python', job.script_path,
            '--',
            '--output', job.output_dir,
            '--taxonomy', self.config['taxonomy'],
            '--count', '1',
            '--categories'
        ] + job.categories

        if job.seed is not None:
            cmd.extend(['--seed', str(job.seed)])

        # Set GPU device if available
        if worker.gpu_id is not None:
            cmd.extend(['--gpu', str(worker.gpu_id)])

        return cmd

    def worker_thread(self, worker: RenderWorker):
        """Worker thread that processes jobs"""
        print(f"[{worker.worker_id}] Worker thread started")

        while self.running:
            try:
                # Get next job from queue (with timeout)
                priority, timestamp, job_id = self.job_queue.get(timeout=1.0)

                job = self.jobs.get(job_id)

                if job and job.status == RenderStatus.PENDING:
                    self.execute_job(job, worker)

                self.job_queue.task_done()

            except queue.Empty:
                # No jobs available, continue waiting
                continue

            except Exception as e:
                print(f"[{worker.worker_id}] Error in worker thread: {e}")

        print(f"[{worker.worker_id}] Worker thread stopped")

    def start(self):
        """Start the orchestrator"""
        print("\n" + "="*60)
        print("Starting Distributed Render Orchestrator")
        print("="*60)

        self.running = True

        # Start worker threads
        threads = []
        for worker in self.workers.values():
            for _ in range(worker.max_concurrent_jobs):
                thread = threading.Thread(target=self.worker_thread, args=(worker,))
                thread.daemon = True
                thread.start()
                threads.append(thread)

        print(f"\n✓ Orchestrator started with {len(threads)} worker threads")

        return threads

    def stop(self):
        """Stop the orchestrator"""
        print("\nStopping orchestrator...")
        self.running = False

    def wait_for_completion(self):
        """Wait for all jobs to complete"""
        self.job_queue.join()

    def get_status(self) -> Dict:
        """Get orchestrator status"""
        status_counts = {s.value: 0 for s in RenderStatus}

        for job in self.jobs.values():
            status_counts[job.status.value] += 1

        worker_statuses = {}
        for worker_id, worker in self.workers.items():
            worker_statuses[worker_id] = {
                'status': worker.status,
                'current_job': worker.current_job,
                'completed': worker.jobs_completed,
                'failed': worker.jobs_failed,
                'total_render_time': worker.total_render_time
            }

        return {
            'jobs': status_counts,
            'workers': worker_statuses,
            'queue_size': self.job_queue.qsize()
        }

    def print_status(self):
        """Print current status"""
        status = self.get_status()

        print("\n" + "="*60)
        print("Render Orchestrator Status")
        print("="*60)

        print("\nJobs:")
        for status_name, count in status['jobs'].items():
            print(f"  {status_name}: {count}")

        print(f"\nQueue size: {status['queue_size']}")

        print("\nWorkers:")
        for worker_id, worker_status in status['workers'].items():
            print(f"  {worker_id}:")
            print(f"    Status: {worker_status['status']}")
            print(f"    Completed: {worker_status['completed']}")
            print(f"    Failed: {worker_status['failed']}")
            print(f"    Total render time: {worker_status['total_render_time']:.2f}s")

    def save_state(self, output_path: Path):
        """Save orchestrator state to file"""
        state = {
            'jobs': {job_id: asdict(job) for job_id, job in self.jobs.items()},
            'workers': {worker_id: asdict(worker) for worker_id, worker in self.workers.items()},
            'timestamp': datetime.utcnow().isoformat()
        }

        # Convert enums to strings
        for job_data in state['jobs'].values():
            job_data['status'] = job_data['status'].value if isinstance(job_data['status'], RenderStatus) else job_data['status']

        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"✓ State saved to {output_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Parallel Synth Distributed Render Orchestrator')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--count', type=int, default=10, help='Number of jobs to create')
    parser.add_argument('--categories', nargs='+', default=['geometry', 'materials', 'lighting', 'camera'],
                       help='Categories to include')
    parser.add_argument('--priority', type=int, default=5, help='Job priority')
    parser.add_argument('--state-file', type=str, default='./orchestrator_state.json',
                       help='Path to save state')

    args = parser.parse_args()

    # Create orchestrator
    config_path = Path(args.config) if args.config else None
    orchestrator = DistributedRenderOrchestrator(config_path)

    # Create jobs
    job_ids = orchestrator.create_batch_jobs(args.categories, args.count, args.priority)

    # Start orchestrator
    threads = orchestrator.start()

    # Monitor progress
    try:
        while orchestrator.job_queue.qsize() > 0 or any(w.status == 'busy' for w in orchestrator.workers.values()):
            time.sleep(5)
            orchestrator.print_status()

        print("\n" + "="*60)
        print("All jobs completed!")
        print("="*60)

        orchestrator.print_status()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        orchestrator.stop()
        orchestrator.save_state(Path(args.state_file))

        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=5)

    return 0


if __name__ == '__main__':
    exit(main())
