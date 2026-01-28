"""
Kafka Integration for Distributed NIDS
Implements message queue-based distributed processing
"""

import json
import logging
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class KafkaNIDSProducer:
    """Sends packets to Kafka for distributed processing"""

    def __init__(self, bootstrap_servers: List[str] = None):
        """
        Initialize Kafka producer

        Args:
            bootstrap_servers: List of Kafka broker addresses
        """
        if bootstrap_servers is None:
            bootstrap_servers = ['localhost:9092']

        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self._initialize_producer()

    def _initialize_producer(self):
        """Initialize Kafka producer"""
        try:
            from kafka import KafkaProducer

            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='snappy',  # Fast compression
                batch_size=16384,           # Batch for efficiency
                linger_ms=10,               # Small delay for batching
                acks=1                      # Wait for leader acknowledgment
            )

            logger.info(f"✅ Kafka producer connected to {self.bootstrap_servers}")

        except ImportError:
            logger.error("kafka-python not installed. Install with: pip install kafka-python")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise

    def send_packet(self, packet: Dict, topic: str = 'network-packets'):
        """
        Send packet to Kafka topic

        Args:
            packet: Packet dictionary
            topic: Kafka topic name
        """
        try:
            # Add metadata
            packet['_kafka_timestamp'] = packet.get('timestamp')

            # Send to Kafka
            future = self.producer.send(topic, value=packet)

            # Optional: Wait for confirmation (slower but more reliable)
            # future.get(timeout=10)

        except Exception as e:
            logger.error(f"Error sending packet to Kafka: {e}")

    def flush(self):
        """Ensure all packets are sent"""
        if self.producer:
            self.producer.flush()

    def close(self):
        """Close producer connection"""
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")


class KafkaNIDSConsumer:
    """Receives packets from Kafka for detection"""

    def __init__(self, bootstrap_servers: List[str] = None,
                 group_id: str = 'nids-detector', worker_id: int = 0):
        """
        Initialize Kafka consumer

        Args:
            bootstrap_servers: List of Kafka broker addresses
            group_id: Consumer group ID for load balancing
            worker_id: Worker identifier
        """
        if bootstrap_servers is None:
            bootstrap_servers = ['localhost:9092']

        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.worker_id = worker_id
        self.consumer = None
        self.alert_producer = None
        self._initialize_consumer()

    def _initialize_consumer(self):
        """Initialize Kafka consumer"""
        try:
            from kafka import KafkaConsumer, KafkaProducer

            self.consumer = KafkaConsumer(
                'network-packets',
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                max_poll_records=500  # Process in batches
            )

            # Producer for alerts
            self.alert_producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='snappy'
            )

            logger.info(f"✅ Kafka consumer {self.worker_id} connected")

        except ImportError:
            logger.error("kafka-python not installed. Install with: pip install kafka-python")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise

    def consume_and_detect(self, detector):
        """
        Consume packets and detect threats

        Args:
            detector: Detection engine with detect() method
        """
        logger.info(f"Worker {self.worker_id} starting consumption...")

        try:
            for message in self.consumer:
                packet = message.value

                # Detect
                try:
                    is_threat, attack_type = detector.detect(packet)

                    if is_threat:
                        self._send_alert(packet, attack_type)

                except Exception as e:
                    logger.error(f"Detection error: {e}")

        except KeyboardInterrupt:
            logger.info(f"Worker {self.worker_id} stopping...")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            self.close()

    def _send_alert(self, packet: Dict, attack_type: str):
        """
        Send alert to alert topic

        Args:
            packet: Packet that triggered alert
            attack_type: Type of attack detected
        """
        alert = {
            'worker_id': self.worker_id,
            'attack_type': attack_type,
            'packet': packet,
            'timestamp': packet.get('timestamp')
        }

        try:
            self.alert_producer.send('nids-alerts', value=alert)
            logger.info(f"🚨 Alert sent: {attack_type} from {packet.get('src_ip')}")
        except Exception as e:
            logger.error(f"Error sending alert: {e}")

    def close(self):
        """Close consumer connection"""
        if self.consumer:
            self.consumer.close()
        if self.alert_producer:
            self.alert_producer.close()
        logger.info(f"Worker {self.worker_id} closed")


class KafkaAlertConsumer:
    """Consumes alerts from Kafka alert topic"""

    def __init__(self, bootstrap_servers: List[str] = None,
                 alert_handler: Optional[Callable] = None):
        """
        Initialize alert consumer

        Args:
            bootstrap_servers: List of Kafka broker addresses
            alert_handler: Function to call for each alert
        """
        if bootstrap_servers is None:
            bootstrap_servers = ['localhost:9092']

        self.bootstrap_servers = bootstrap_servers
        self.alert_handler = alert_handler
        self.consumer = None
        self._initialize_consumer()

    def _initialize_consumer(self):
        """Initialize Kafka consumer for alerts"""
        try:
            from kafka import KafkaConsumer

            self.consumer = KafkaConsumer(
                'nids-alerts',
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id='alert-processor'
            )

            logger.info("✅ Alert consumer connected")

        except ImportError:
            logger.error("kafka-python not installed. Install with: pip install kafka-python")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize alert consumer: {e}")
            raise

    def consume_alerts(self):
        """Consume and process alerts"""
        logger.info("Alert consumer starting...")

        try:
            for message in self.consumer:
                alert = message.value

                if self.alert_handler:
                    try:
                        self.alert_handler(alert)
                    except Exception as e:
                        logger.error(f"Alert handler error: {e}")
                else:
                    print(f"🚨 ALERT: {alert['attack_type']} from "
                          f"{alert['packet'].get('src_ip')} "
                          f"(Worker {alert['worker_id']})")

        except KeyboardInterrupt:
            logger.info("Alert consumer stopping...")
        except Exception as e:
            logger.error(f"Alert consumer error: {e}")
        finally:
            self.close()

    def close(self):
        """Close consumer connection"""
        if self.consumer:
            self.consumer.close()
        logger.info("Alert consumer closed")


# Example usage
def example_producer():
    """Example: Send packets to Kafka"""
    producer = KafkaNIDSProducer()

    # Send 100 test packets
    for i in range(100):
        packet = {
            'src_ip': f'192.168.1.{i % 255}',
            'dst_ip': '10.0.0.5',
            'protocol': 'TCP',
            'packet_size': 1500,
            'src_port': 50000 + i,
            'dst_port': 80,
            'timestamp': str(i)
        }

        producer.send_packet(packet)

        if i % 10 == 0:
            print(f"Sent {i} packets...")

    producer.flush()
    producer.close()
    print("Producer complete!")


def example_consumer():
    """Example: Consume packets and detect"""
    from main import HybridDetectionEngine, MLAnomalyDetectionEngine, SignatureDetectionEngine

    # Create detector
    ml_engine = MLAnomalyDetectionEngine()
    sig_engine = SignatureDetectionEngine()
    detector = HybridDetectionEngine(sig_engine, ml_engine)

    # Create consumer
    consumer = KafkaNIDSConsumer(worker_id=1)

    # Consume and detect
    consumer.consume_and_detect(detector)


def example_alert_consumer():
    """Example: Consume and display alerts"""

    def handle_alert(alert):
        print(f"\n{'='*60}")
        print(f"🚨 SECURITY ALERT")
        print(f"{'='*60}")
        print(f"Attack Type: {alert['attack_type']}")
        print(f"Source IP: {alert['packet'].get('src_ip')}")
        print(f"Destination IP: {alert['packet'].get('dst_ip')}")
        print(f"Worker ID: {alert['worker_id']}")
        print(f"{'='*60}\n")

    consumer = KafkaAlertConsumer(alert_handler=handle_alert)
    consumer.consume_alerts()


if __name__ == "__main__":
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == 'producer':
            print("Running Kafka Producer...")
            example_producer()
        elif mode == 'consumer':
            print("Running Kafka Consumer...")
            example_consumer()
        elif mode == 'alerts':
            print("Running Alert Consumer...")
            example_alert_consumer()
        else:
            print("Usage: python kafka_integration.py [producer|consumer|alerts]")
    else:
        print("Usage: python kafka_integration.py [producer|consumer|alerts]")
