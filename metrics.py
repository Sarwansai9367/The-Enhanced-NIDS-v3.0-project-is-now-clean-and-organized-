"""
Prometheus Metrics for NIDS
Exports real-time metrics for monitoring and alerting
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class NIDSMetrics:
    """Export metrics to Prometheus"""

    def __init__(self, port: int = 9090):
        """
        Initialize metrics exporter

        Args:
            port: Port to expose metrics on
        """
        self.port = port
        self.metrics_enabled = False

        # Try to import prometheus_client
        try:
            from prometheus_client import Counter, Histogram, Gauge, start_http_server

            # Counters (cumulative)
            self.packets_processed = Counter(
                'nids_packets_processed_total',
                'Total number of packets processed'
            )

            self.attacks_detected = Counter(
                'nids_attacks_detected_total',
                'Total number of attacks detected',
                ['attack_type']
            )

            self.alerts_sent = Counter(
                'nids_alerts_sent_total',
                'Total number of alerts sent',
                ['severity', 'channel']
            )

            self.false_positives = Counter(
                'nids_false_positives_total',
                'Total number of false positives'
            )

            # Gauges (current value)
            self.queue_size = Gauge(
                'nids_queue_size',
                'Current packet queue size'
            )

            self.detection_rate = Gauge(
                'nids_detection_rate_percent',
                'Current attack detection rate (%)'
            )

            self.active_connections = Gauge(
                'nids_active_connections',
                'Number of active connections being tracked'
            )

            self.blocked_ips = Gauge(
                'nids_blocked_ips',
                'Number of currently blocked IPs'
            )

            # Histograms (distributions)
            self.packet_processing_time = Histogram(
                'nids_packet_processing_seconds',
                'Time to process a single packet',
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
            )

            self.alert_latency = Histogram(
                'nids_alert_latency_seconds',
                'Time from detection to alert sent',
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            )

            self.ml_inference_time = Histogram(
                'nids_ml_inference_seconds',
                'ML model inference time',
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
            )

            self.metrics_enabled = True
            logger.info(f"✅ Prometheus metrics initialized")

        except ImportError:
            logger.warning("prometheus_client not installed. Metrics disabled.")
            logger.warning("Install with: pip install prometheus-client")

    def start_server(self):
        """Start Prometheus metrics server"""
        if not self.metrics_enabled:
            logger.warning("Metrics not enabled, server not started")
            return

        try:
            from prometheus_client import start_http_server

            start_http_server(self.port)
            logger.info(f"✅ Metrics server started on port {self.port}")
            logger.info(f"   Metrics available at http://localhost:{self.port}/metrics")

        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    def record_packet_processed(self):
        """Record a processed packet"""
        if self.metrics_enabled:
            self.packets_processed.inc()

    def record_attack_detected(self, attack_type: str):
        """
        Record an attack detection

        Args:
            attack_type: Type of attack detected
        """
        if self.metrics_enabled:
            self.attacks_detected.labels(attack_type=attack_type).inc()

    def record_alert_sent(self, severity: str, channel: str):
        """
        Record an alert sent

        Args:
            severity: Alert severity (CRITICAL, HIGH, MEDIUM, LOW)
            channel: Alert channel (email, sms, webhook, etc.)
        """
        if self.metrics_enabled:
            self.alerts_sent.labels(severity=severity, channel=channel).inc()

    def record_false_positive(self):
        """Record a false positive"""
        if self.metrics_enabled:
            self.false_positives.inc()

    def update_queue_size(self, size: int):
        """
        Update current queue size

        Args:
            size: Current queue size
        """
        if self.metrics_enabled:
            self.queue_size.set(size)

    def update_detection_rate(self, rate: float):
        """
        Update detection rate

        Args:
            rate: Detection rate as percentage (0-100)
        """
        if self.metrics_enabled:
            self.detection_rate.set(rate)

    def update_active_connections(self, count: int):
        """
        Update active connections count

        Args:
            count: Number of active connections
        """
        if self.metrics_enabled:
            self.active_connections.set(count)

    def update_blocked_ips(self, count: int):
        """
        Update blocked IPs count

        Args:
            count: Number of blocked IPs
        """
        if self.metrics_enabled:
            self.blocked_ips.set(count)

    def observe_packet_processing_time(self, duration: float):
        """
        Record packet processing time

        Args:
            duration: Processing time in seconds
        """
        if self.metrics_enabled:
            self.packet_processing_time.observe(duration)

    def observe_alert_latency(self, duration: float):
        """
        Record alert latency

        Args:
            duration: Latency in seconds
        """
        if self.metrics_enabled:
            self.alert_latency.observe(duration)

    def observe_ml_inference_time(self, duration: float):
        """
        Record ML inference time

        Args:
            duration: Inference time in seconds
        """
        if self.metrics_enabled:
            self.ml_inference_time.observe(duration)


# Global metrics instance
metrics = NIDSMetrics()


# Example usage
def example_usage():
    """Example of using metrics"""
    import time
    import random

    # Start metrics server
    metrics.start_server()

    print(f"\n✅ Metrics server running on http://localhost:{metrics.port}/metrics")
    print("\nSimulating NIDS activity...\n")

    attack_types = ['DDoS', 'Port Scan', 'Malware', 'Brute Force', 'SQL Injection']
    severities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    channels = ['email', 'sms', 'webhook', 'console']

    try:
        for i in range(100):
            # Simulate packet processing
            start = time.time()
            time.sleep(random.uniform(0.001, 0.01))  # Simulate processing
            duration = time.time() - start

            metrics.record_packet_processed()
            metrics.observe_packet_processing_time(duration)

            # Randomly detect attacks
            if random.random() < 0.1:  # 10% attack rate
                attack_type = random.choice(attack_types)
                metrics.record_attack_detected(attack_type)

                # Send alert
                severity = random.choice(severities)
                channel = random.choice(channels)
                metrics.record_alert_sent(severity, channel)

                # Record alert latency
                metrics.observe_alert_latency(random.uniform(0.01, 0.5))

                print(f"🚨 Attack detected: {attack_type} (Severity: {severity})")

            # Update gauges
            metrics.update_queue_size(random.randint(0, 1000))
            metrics.update_detection_rate(random.uniform(95, 99.9))
            metrics.update_active_connections(random.randint(100, 500))
            metrics.update_blocked_ips(random.randint(0, 50))

            # ML inference time
            metrics.observe_ml_inference_time(random.uniform(0.001, 0.05))

            if i % 10 == 0:
                print(f"Processed {i} packets...")

            time.sleep(0.1)

        print("\n✅ Simulation complete!")
        print(f"\nMetrics available at: http://localhost:{metrics.port}/metrics")
        print("Press Ctrl+C to stop the metrics server")

        # Keep server running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopping metrics server...")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=== NIDS Prometheus Metrics ===")
    example_usage()
