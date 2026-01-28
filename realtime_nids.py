"""
Complete Real-Time Network Intrusion Detection System
Integrates packet capture, detection, logging, and alerting
"""

import threading
import time
import queue
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import logging
import sys
import os

# Import real-time modules
from realtime_capture import RealTimePacketCapture, StreamingDetector
from realtime_logger import RealTimeLogger
from realtime_notifier import AlertNotifier

# Import main NIDS modules
from main import (
    FeatureExtractionModule,
    PreprocessingModule,
    SignatureDetectionEngine,
    MLAnomalyDetectionEngine,
    HybridDetectionEngine
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('nids_realtime.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RealTimeNIDS:
    """
    Complete Real-Time Network Intrusion Detection System
    Combines all modules for production-ready threat detection
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Real-Time NIDS

        Args:
            config: Configuration dictionary
        """
        logger.info("="*70)
        logger.info("INITIALIZING REAL-TIME NETWORK INTRUSION DETECTION SYSTEM")
        logger.info("="*70)

        self.config = config or self._default_config()

        # Initialize components
        self.packet_capture = RealTimePacketCapture(
            max_queue_size=self.config['capture']['queue_size']
        )
        self.stream_detector = StreamingDetector(
            window_size=self.config['detection']['window_size'],
            window_time=self.config['detection']['window_time']
        )
        self.feature_extractor = FeatureExtractionModule()
        self.preprocessor = PreprocessingModule()
        self.signature_engine = SignatureDetectionEngine()
        self.ml_engine = MLAnomalyDetectionEngine(
            model_type=self.config['ml']['model_type']
        )
        self.hybrid_engine = None  # Initialized after training
        self.logger_db = RealTimeLogger(
            db_path=self.config['logging']['db_path'],
            batch_size=self.config['logging']['batch_size']
        )
        self.notifier = AlertNotifier(
            config_file=self.config['alerting'].get('config_file')
        )

        # Runtime state
        self.is_running = False
        self.detection_thread = None
        self.analysis_thread = None

        # Statistics
        self.stats = {
            'packets_processed': 0,
            'intrusions_detected': 0,
            'alerts_generated': 0,
            'start_time': None,
            'ml_predictions': 0,
            'signature_detections': 0
        }

        logger.info("✅ All components initialized successfully")

    def _default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'capture': {
                'queue_size': 10000,
                'interface': None,  # None = all interfaces
                'filter_bpf': None,  # BPF filter string
            },
            'detection': {
                'window_size': 100,
                'window_time': 60,  # seconds
                'enable_ml': True,
                'enable_signature': True,
            },
            'ml': {
                'model_type': 'random_forest',
                'model_path': None,
            },
            'logging': {
                'db_path': 'nids_realtime.db',
                'batch_size': 100,
            },
            'alerting': {
                'config_file': None,
                'min_severity': 'LOW',
            },
            'performance': {
                'num_detection_threads': 2,
                'stats_interval': 10,  # seconds
            }
        }

    def train_ml_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the ML detection model

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("Training ML detection model...")

        # Normalize features
        X_train_scaled, _ = self.preprocessor.normalize_features(X_train)

        # Train model
        self.ml_engine.train(X_train_scaled, y_train)

        # Initialize hybrid engine
        self.hybrid_engine = HybridDetectionEngine(
            self.signature_engine,
            self.ml_engine
        )

        logger.info("✅ ML model trained and hybrid engine initialized")

    def load_ml_model(self, model_path: str):
        """
        Load pre-trained ML model

        Args:
            model_path: Path to saved model file
        """
        import joblib
        logger.info(f"Loading model from {model_path}")

        try:
            # Load the saved model
            self.ml_engine.model = joblib.load(model_path)

            # Initialize hybrid engine
            self.hybrid_engine = HybridDetectionEngine(
                self.signature_engine,
                self.ml_engine
            )

            logger.info("✅ ML model loaded successfully")
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def start(self, use_real_capture: bool = True, duration: Optional[int] = None):
        """
        Start real-time detection system

        Args:
            use_real_capture: Use real packet capture (requires Scapy and permissions)
            duration: Run duration in seconds (None = unlimited)
        """
        if self.is_running:
            logger.warning("System already running")
            return

        if not self.hybrid_engine:
            logger.error("ML model not trained! Call train_ml_model() first")
            return

        logger.info("="*70)
        logger.info("STARTING REAL-TIME DETECTION")
        logger.info("="*70)
        logger.info(f"Mode: {'REAL CAPTURE' if use_real_capture else 'SIMULATION'}")
        logger.info(f"Duration: {duration if duration else 'UNLIMITED'} seconds")

        self.is_running = True
        self.stats['start_time'] = datetime.now()

        # Start database logger
        self.logger_db.start_logging()

        # Start packet capture
        if use_real_capture and self.packet_capture.scapy_available:
            self.packet_capture.start_capture(
                interface=self.config['capture']['interface'],
                filter_bpf=self.config['capture']['filter_bpf']
            )
        else:
            logger.warning("Real capture not available, using simulation")
            # Start simulation thread
            threading.Thread(target=self._simulate_traffic, daemon=True).start()

        # Start detection threads
        for i in range(self.config['performance']['num_detection_threads']):
            thread = threading.Thread(
                target=self._detection_worker,
                name=f"DetectionWorker-{i}",
                daemon=True
            )
            thread.start()

        # Start analysis thread for pattern detection
        self.analysis_thread = threading.Thread(
            target=self._analysis_worker,
            daemon=True
        )
        self.analysis_thread.start()

        # Start statistics reporter
        threading.Thread(target=self._stats_reporter, daemon=True).start()

        logger.info("🚀 Real-time detection started successfully")

        # Run for specified duration
        if duration:
            logger.info(f"System will run for {duration} seconds...")
            time.sleep(duration)
            self.stop()
        else:
            logger.info("System running indefinitely. Press Ctrl+C to stop.")

    def stop(self):
        """Stop real-time detection system"""
        if not self.is_running:
            return

        logger.info("Stopping real-time detection system...")
        self.is_running = False

        # Stop capture
        self.packet_capture.stop_capture()

        # Stop logger
        self.logger_db.stop_logging()

        # Print final statistics
        self._print_final_stats()

        logger.info("✅ System stopped successfully")

    def _detection_worker(self):
        """Worker thread for packet detection"""
        logger.info(f"Detection worker {threading.current_thread().name} started")

        while self.is_running:
            try:
                # Get packet from queue
                packet = self.packet_capture.get_packet(timeout=1.0)

                if packet is None:
                    continue

                # Process packet
                self._process_packet(packet)

            except Exception as e:
                logger.error(f"Detection worker error: {e}")

        logger.info(f"Detection worker {threading.current_thread().name} stopped")

    def _process_packet(self, packet: Dict):
        """
        Process single packet through detection pipeline

        Args:
            packet: Packet dictionary
        """
        try:
            # Extract features
            packet_features = self.feature_extractor.extract_features(packet)

            # Prepare ML features
            ml_features = self._prepare_ml_features(packet_features)

            # Hybrid detection
            detection_result = self.hybrid_engine.detect(packet_features, ml_features)

            # Update statistics
            self.stats['packets_processed'] += 1
            if detection_result['is_intrusion']:
                self.stats['intrusions_detected'] += 1

            if 'Signature' in detection_result.get('detection_method', []):
                self.stats['signature_detections'] += 1
            if 'ML' in detection_result.get('detection_method', []):
                self.stats['ml_predictions'] += 1

            # Log to database
            self.logger_db.log_packet(packet, detection_result)

            # Generate alert if intrusion detected
            if detection_result['is_intrusion']:
                self._generate_alert(packet, detection_result)

            # Add to streaming detector for pattern analysis
            self.stream_detector.add_packet(packet)

        except Exception as e:
            logger.error(f"Packet processing error: {e}")

    def _prepare_ml_features(self, packet_features: Dict) -> np.ndarray:
        """
        Prepare feature vector for ML model

        Args:
            packet_features: Extracted packet features

        Returns:
            Feature vector as numpy array
        """
        features = [
            packet_features.get('protocol_type', 0),
            packet_features.get('packet_size', 0),
            packet_features.get('src_port', 0),
            packet_features.get('dst_port', 0),
            packet_features.get('flag_syn', 0),
            packet_features.get('flag_ack', 0),
            packet_features.get('flow_duration', 0),
            packet_features.get('flow_packet_count', 0),
        ]

        ml_features = np.array(features)

        # Ensure correct dimensions for model
        if hasattr(self.ml_engine, 'model') and self.ml_engine.model is not None:
            expected_features = self.ml_engine.model.n_features_in_
            if len(ml_features) < expected_features:
                ml_features = np.pad(ml_features, (0, expected_features - len(ml_features)))
            elif len(ml_features) > expected_features:
                ml_features = ml_features[:expected_features]

        return ml_features

    def _generate_alert(self, packet: Dict, detection_result: Dict):
        """
        Generate and send alert

        Args:
            packet: Packet information
            detection_result: Detection result
        """
        alert_data = {
            'timestamp': datetime.now(),
            'severity': self._calculate_severity(detection_result['attack_type']),
            'attack_type': detection_result['attack_type'],
            'source_ip': packet.get('src_ip', 'Unknown'),
            'destination_ip': packet.get('dst_ip', 'Unknown'),
            'protocol': packet.get('protocol', 'Unknown'),
            'detection_method': ', '.join(detection_result['detection_method']),
            'confidence': detection_result['confidence'],
            'description': f"Intrusion detected: {detection_result['attack_type']}"
        }

        # Log alert to database
        self.logger_db.log_alert(alert_data)

        # Send notifications
        self.notifier.send_alert(alert_data, priority=alert_data['severity'])

        self.stats['alerts_generated'] += 1

    def _calculate_severity(self, attack_type: str) -> str:
        """Calculate alert severity based on attack type"""
        critical_keywords = ['malware', 'ransomware', 'smb', 'eternalblue', 'shellshock']
        high_keywords = ['ddos', 'syn flood', 'brute force', 'sql injection', 'exploit']
        medium_keywords = ['port scan', 'sweep', 'suspicious']

        attack_lower = attack_type.lower()

        for keyword in critical_keywords:
            if keyword in attack_lower:
                return 'CRITICAL'

        for keyword in high_keywords:
            if keyword in attack_lower:
                return 'HIGH'

        for keyword in medium_keywords:
            if keyword in attack_lower:
                return 'MEDIUM'

        return 'LOW'

    def _analysis_worker(self):
        """Worker for analyzing packet patterns in sliding window"""
        logger.info("Pattern analysis worker started")

        while self.is_running:
            try:
                time.sleep(5)  # Analyze every 5 seconds

                # Analyze window for attack patterns
                attack = self.stream_detector.analyze_window()

                if attack:
                    logger.warning(f"Pattern detected: {attack['type']} - {attack['severity']}")

                    # Generate pattern-based alert
                    alert_data = {
                        'timestamp': datetime.now(),
                        'severity': attack['severity'],
                        'attack_type': attack['type'],
                        'source_ip': 'Multiple',
                        'destination_ip': 'Multiple',
                        'protocol': 'Multiple',
                        'detection_method': 'Pattern Analysis',
                        'confidence': attack.get('confidence', 0.85),
                        'description': attack.get('details', 'Pattern-based detection')
                    }

                    self.logger_db.log_alert(alert_data)
                    self.notifier.send_alert(alert_data, priority=attack['severity'])

            except Exception as e:
                logger.error(f"Analysis worker error: {e}")

        logger.info("Pattern analysis worker stopped")

    def _simulate_traffic(self):
        """Simulate network traffic for testing"""
        logger.info("Starting traffic simulation...")

        import random
        protocols = ['TCP', 'UDP', 'ICMP']

        while self.is_running:
            packet = {
                'timestamp': datetime.now(),
                'src_ip': f'192.168.1.{random.randint(1, 254)}',
                'dst_ip': f'10.0.0.{random.randint(1, 254)}',
                'protocol': random.choice(protocols),
                'packet_size': random.randint(64, 1500),
                'src_port': random.randint(1024, 65535),
                'dst_port': random.randint(1, 1024),
                'flags': random.choice(['SYN', 'ACK', 'SYN-ACK', ''])
            }

            try:
                self.packet_capture.packet_queue.put_nowait(packet)
            except queue.Full:
                pass

            time.sleep(0.1)  # 10 packets per second

    def _stats_reporter(self):
        """Periodically report statistics"""
        while self.is_running:
            time.sleep(self.config['performance']['stats_interval'])
            self._print_stats()

    def _print_stats(self):
        """Print current statistics"""
        runtime = (datetime.now() - self.stats['start_time']).total_seconds()
        pps = self.stats['packets_processed'] / runtime if runtime > 0 else 0

        logger.info("="*70)
        logger.info("REAL-TIME STATISTICS")
        logger.info("="*70)
        logger.info(f"Runtime:             {runtime:.1f}s")
        logger.info(f"Packets Processed:   {self.stats['packets_processed']}")
        logger.info(f"Processing Rate:     {pps:.2f} packets/sec")
        logger.info(f"Intrusions Detected: {self.stats['intrusions_detected']}")
        logger.info(f"Detection Rate:      {(self.stats['intrusions_detected']/max(1,self.stats['packets_processed'])*100):.1f}%")
        logger.info(f"Alerts Generated:    {self.stats['alerts_generated']}")
        logger.info(f"Signature Detections: {self.stats['signature_detections']}")
        logger.info(f"ML Predictions:      {self.stats['ml_predictions']}")

        # Capture stats
        capture_stats = self.packet_capture.get_stats()
        logger.info(f"Queue Size:          {capture_stats['queue_size']}")
        logger.info(f"Packets Dropped:     {capture_stats['packets_dropped']}")
        logger.info("="*70)

    def _print_final_stats(self):
        """Print final statistics"""
        logger.info("\n" + "="*70)
        logger.info("FINAL SYSTEM STATISTICS")
        logger.info("="*70)

        runtime = (datetime.now() - self.stats['start_time']).total_seconds()

        logger.info(f"Total Runtime:       {runtime:.1f}s")
        logger.info(f"Packets Processed:   {self.stats['packets_processed']}")
        logger.info(f"Avg Processing Rate: {self.stats['packets_processed']/runtime:.2f} pps")
        logger.info(f"Intrusions Detected: {self.stats['intrusions_detected']}")
        logger.info(f"Detection Rate:      {(self.stats['intrusions_detected']/max(1,self.stats['packets_processed'])*100):.1f}%")
        logger.info(f"Alerts Generated:    {self.stats['alerts_generated']}")

        # Database stats
        db_stats = self.logger_db.get_statistics('1 hour')
        logger.info(f"\nDatabase Statistics:")
        logger.info(f"  Packets Logged:    {db_stats['packets_logged']}")
        logger.info(f"  Alerts Logged:     {db_stats['alerts_logged']}")

        # Notification stats
        notif_stats = self.notifier.get_notification_stats()
        logger.info(f"\nNotification Statistics:")
        logger.info(f"  Total Sent:        {notif_stats['total_notifications']}")
        logger.info(f"  By Channel:        {notif_stats.get('by_channel', {})}")

        logger.info("="*70 + "\n")


# Example usage
if __name__ == '__main__':
    print("="*70)
    print("REAL-TIME NETWORK INTRUSION DETECTION SYSTEM")
    print("="*70)

    try:
        # Initialize system
        nids = RealTimeNIDS()

        # Load and train model (simplified for example)
        print("\n[*] Training ML model...")
        from sklearn.datasets import make_classification
        X_train, y_train = make_classification(
            n_samples=1000,
            n_features=8,
            n_classes=2,
            random_state=42
        )
        nids.train_ml_model(X_train, y_train)

        print("\n[*] Starting real-time detection...")
        print("[*] System will run for 60 seconds...")
        print("[*] Press Ctrl+C to stop early\n")

        # Start system (simulation mode for safety)
        nids.start(use_real_capture=False, duration=60)

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        if nids.is_running:
            nids.stop()

    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n✅ Real-Time NIDS execution completed")
