"""
Parallel NIDS - Multi-core parallel detection system
Implements multi-processing for maximum throughput
"""

import multiprocessing as mp
from multiprocessing import Queue, Process
import numpy as np
import logging
from typing import Dict, Optional
import time

logger = logging.getLogger(__name__)


class ParallelNIDS:
    """Multi-core parallel detection system"""

    def __init__(self, num_workers: Optional[int] = None):
        """
        Initialize parallel NIDS

        Args:
            num_workers: Number of worker processes (default: CPU count)
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.packet_queue = Queue(maxsize=10000)
        self.alert_queue = Queue(maxsize=1000)
        self.workers = []
        self.is_running = False

    def start(self):
        """Start worker processes"""
        logger.info(f"Starting {self.num_workers} worker processes")
        self.is_running = True

        for i in range(self.num_workers):
            worker = Process(
                target=self._worker_process,
                args=(i, self.packet_queue, self.alert_queue)
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

        logger.info(f"✅ {self.num_workers} workers started")

    def _worker_process(self, worker_id: int, packet_queue: Queue, alert_queue: Queue):
        """
        Worker process - detects threats

        Args:
            worker_id: Worker identifier
            packet_queue: Input queue for packets
            alert_queue: Output queue for alerts
        """
        # Import here to avoid pickling issues
        import sys
        import os

        # Add parent directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        try:
            from main import MLAnomalyDetectionEngine, SignatureDetectionEngine
            import numpy as np

            # Each worker has its own ML model
            ml_engine = MLAnomalyDetectionEngine()
            sig_engine = SignatureDetectionEngine()

            logger.info(f"Worker {worker_id} initialized")

            while True:
                try:
                    # Get packet from queue
                    packet = packet_queue.get(timeout=1)

                    if packet is None:  # Poison pill
                        break

                    # Detect
                    is_threat, attack_type = self._detect(
                        packet, ml_engine, sig_engine
                    )

                    if is_threat:
                        alert_queue.put({
                            'worker_id': worker_id,
                            'packet': packet,
                            'attack_type': attack_type,
                            'timestamp': time.time()
                        })

                except Exception as e:
                    if self.is_running:
                        logger.debug(f"Worker {worker_id} error: {e}")
                    continue

        except Exception as e:
            logger.error(f"Worker {worker_id} failed to initialize: {e}")

    def _detect(self, packet: Dict, ml_engine, sig_engine) -> tuple:
        """
        Detect threats in packet

        Args:
            packet: Packet dictionary
            ml_engine: ML detection engine
            sig_engine: Signature detection engine

        Returns:
            Tuple of (is_threat, attack_type)
        """
        try:
            # Signature detection
            sig_threat = sig_engine.detect(packet)
            if sig_threat:
                return True, sig_threat

            # ML detection (if signature didn't find anything)
            # Convert packet to features
            features = self._extract_ml_features(packet)
            if features is not None:
                prediction = ml_engine.predict(features.reshape(1, -1))
                if prediction[0] == 1:
                    return True, "ML_ANOMALY"

            return False, None

        except Exception as e:
            logger.debug(f"Detection error: {e}")
            return False, None

    def _extract_ml_features(self, packet: Dict) -> Optional[np.ndarray]:
        """
        Extract ML features from packet

        Args:
            packet: Packet dictionary

        Returns:
            Feature array or None if extraction fails
        """
        try:
            # Extract 11 features for ML model
            protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2}

            features = np.array([
                1.0,  # duration (placeholder)
                packet.get('packet_size', 0),
                protocol_map.get(packet.get('protocol', 'OTHER'), 3),
                0,  # service (placeholder)
                0,  # flag (placeholder)
                packet.get('src_port', 0),
                packet.get('dst_port', 0),
                1,  # count
                1,  # srv_count
                0,  # serror_rate
                0   # srv_serror_rate
            ])

            return features
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None

    def feed_packet(self, packet: Dict):
        """
        Add packet to processing queue

        Args:
            packet: Packet dictionary
        """
        try:
            self.packet_queue.put(packet, block=False)
        except Exception:
            # Queue full, drop packet
            logger.debug("Packet queue full, dropping packet")

    def get_alert(self, timeout: float = 1) -> Optional[Dict]:
        """
        Get alert from any worker

        Args:
            timeout: Timeout in seconds

        Returns:
            Alert dictionary or None if no alert available
        """
        try:
            return self.alert_queue.get(timeout=timeout)
        except Exception:
            return None

    def stop(self):
        """Stop all workers"""
        logger.info("Stopping workers...")
        self.is_running = False

        # Send poison pills
        for _ in range(self.num_workers):
            self.packet_queue.put(None)

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        logger.info("All workers stopped")

    def get_queue_size(self) -> int:
        """Get current packet queue size"""
        return self.packet_queue.qsize()


# Usage example
def example_usage():
    """Example of using parallel NIDS"""

    # Create parallel NIDS with 8 workers
    nids = ParallelNIDS(num_workers=8)
    nids.start()

    # Simulate packet stream
    for i in range(1000):
        packet = {
            'src_ip': f'192.168.1.{i % 255}',
            'dst_ip': '10.0.0.5',
            'protocol': 'TCP',
            'packet_size': 1500,
            'src_port': 50000 + i,
            'dst_port': 80
        }

        nids.feed_packet(packet)

        # Check for alerts
        alert = nids.get_alert(timeout=0.01)
        if alert:
            print(f"🚨 ALERT from Worker {alert['worker_id']}: "
                  f"{alert['attack_type']} - {alert['packet']['src_ip']}")

    # Stop workers
    nids.stop()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Starting Parallel NIDS Example...")
    example_usage()
    print("Example complete!")
