"""
GPU Monitor - Background nvidia-smi polling for utilization, power, and memory.
Runs as a daemon thread that does not interfere with inference timing.
"""

import subprocess
import threading
import time


class GPUMonitor:
    def __init__(self, poll_interval_ms: int = 200):
        self.poll_interval = poll_interval_ms / 1000.0
        self.samples = []
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def _poll_loop(self):
        while self._running:
            try:
                result = subprocess.run(
                    ['nvidia-smi',
                     '--query-gpu=utilization.gpu,power.draw,memory.used',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(', ')
                    if len(parts) == 3:
                        sample = {
                            'timestamp': time.time(),
                            'gpu_util_pct': float(parts[0]),
                            'power_draw_w': float(parts[1]),
                            'memory_used_mb': float(parts[2]),
                        }
                        with self._lock:
                            self.samples.append(sample)
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def start(self):
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        with self._lock:
            return list(self.samples)

    def get_summary(self):
        with self._lock:
            if not self.samples:
                return {'gpu_util_mean_pct': 0, 'power_draw_mean_w': 0}
            utils = [s['gpu_util_pct'] for s in self.samples]
            powers = [s['power_draw_w'] for s in self.samples]
            return {
                'gpu_util_mean_pct': sum(utils) / len(utils),
                'power_draw_mean_w': sum(powers) / len(powers),
            }
