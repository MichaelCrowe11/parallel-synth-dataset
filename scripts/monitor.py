#!/usr/bin/env python3
"""
Parallel Synth - Real-time Monitoring Dashboard
Monitors generation progress, quality metrics, and costs
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import sys
from typing import Dict, List

class ParallelSynthMonitor:
    """Real-time monitoring for dataset generation"""

    def __init__(self, samples_dir: Path, target: int = 500_000_000):
        self.samples_dir = Path(samples_dir)
        self.target = target
        self.start_time = datetime.now()

    def count_samples(self) -> int:
        """Count generated samples"""
        if not self.samples_dir.exists():
            return 0
        return len([d for d in self.samples_dir.iterdir() if d.is_dir()])

    def load_validation_report(self) -> Dict:
        """Load latest validation report"""
        reports_dir = Path('./output/reports')
        if not reports_dir.exists():
            return {}

        report_files = list(reports_dir.glob('*.json'))
        if not report_files:
            return {}

        # Get most recent report
        latest = max(report_files, key=lambda p: p.stat().st_mtime)

        with open(latest) as f:
            return json.load(f)

    def calculate_rate(self, count: int) -> float:
        """Calculate samples per hour"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed < 60:  # Less than 1 minute
            return 0
        return (count / elapsed) * 3600  # per hour

    def estimate_completion(self, count: int, rate: float) -> datetime:
        """Estimate completion time"""
        if rate == 0:
            return None

        remaining = self.target - count
        hours_remaining = remaining / rate
        return datetime.now() + timedelta(hours=hours_remaining)

    def estimate_cost(self, count: int, cost_per_sample: float = 0.70) -> float:
        """Estimate total cost"""
        return count * cost_per_sample

    def get_storage_size(self) -> float:
        """Get total storage size in GB"""
        if not self.samples_dir.exists():
            return 0

        total_size = 0
        for item in self.samples_dir.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size

        return total_size / (1024 ** 3)  # Convert to GB

    def display_dashboard(self):
        """Display monitoring dashboard"""
        # Clear screen
        print("\033[2J\033[H")

        count = self.count_samples()
        rate = self.calculate_rate(count)
        completion = self.estimate_completion(count, rate)
        cost = self.estimate_cost(count)
        storage = self.get_storage_size()

        # Load validation report
        validation = self.load_validation_report()
        quality_score = validation.get('summary', {}).get('average_quality_score', 0)
        validation_rate = validation.get('summary', {}).get('validation_rate', 0)

        print("="*80)
        print("  PARALLEL SYNTH - GENERATION MONITOR".center(80))
        print("="*80)
        print()

        # Progress
        progress = (count / self.target) * 100
        bar_width = 50
        filled = int(bar_width * progress / 100)
        bar = '█' * filled + '░' * (bar_width - filled)

        print(f"Progress: [{bar}] {progress:.2f}%")
        print(f"Samples:  {count:,} / {self.target:,}")
        print()

        # Metrics
        print("="*80)
        print("  CURRENT METRICS")
        print("="*80)
        print(f"  Generation Rate:    {rate:.1f} samples/hour")
        print(f"  Quality Score:      {quality_score:.3f}")
        print(f"  Validation Rate:    {validation_rate*100:.1f}%")
        print(f"  Storage Used:       {storage:.2f} GB")
        print()

        # Cost
        print("="*80)
        print("  COST ESTIMATES")
        print("="*80)
        print(f"  Cost So Far:        ${cost:,.2f}")
        print(f"  Total Estimated:    ${self.estimate_cost(self.target):,.2f}")
        print()

        # Timeline
        print("="*80)
        print("  TIMELINE")
        print("="*80)
        print(f"  Started:            {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if completion:
            print(f"  Est. Completion:    {completion.strftime('%Y-%m-%d %H:%M:%S')}")

            days_remaining = (completion - datetime.now()).days
            print(f"  Days Remaining:     {days_remaining}")
        else:
            print(f"  Est. Completion:    Calculating...")

        print()

        # Category breakdown (if available)
        if validation and 'categories' in validation.get('summary', {}):
            print("="*80)
            print("  CATEGORY DISTRIBUTION")
            print("="*80)

            categories = validation['summary']['categories']
            for category, cat_count in sorted(categories.items()):
                percentage = (cat_count / count * 100) if count > 0 else 0
                print(f"  {category:20s}  {cat_count:8,} ({percentage:5.1f}%)")

            print()

        # Footer
        print("="*80)
        print(f"  Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Press Ctrl+C to exit")
        print("="*80)

    def run(self, refresh_interval: int = 5):
        """Run monitoring dashboard"""
        print("Starting Parallel Synth Monitor...")
        print(f"Monitoring directory: {self.samples_dir}")
        print(f"Refresh interval: {refresh_interval} seconds")
        print()

        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\n\nMonitor stopped by user")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Parallel Synth Generation Monitor')
    parser.add_argument('--samples-dir', type=str, default='./output/samples',
                       help='Samples directory to monitor')
    parser.add_argument('--target', type=int, default=500_000_000,
                       help='Target number of samples')
    parser.add_argument('--refresh', type=int, default=5,
                       help='Refresh interval in seconds')

    args = parser.parse_args()

    monitor = ParallelSynthMonitor(
        samples_dir=Path(args.samples_dir),
        target=args.target
    )

    monitor.run(refresh_interval=args.refresh)

if __name__ == '__main__':
    main()
