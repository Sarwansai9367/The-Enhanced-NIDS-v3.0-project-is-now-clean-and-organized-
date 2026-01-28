"""
Redis State Management for Real-Time NIDS
Implements ultra-fast state tracking and pattern detection
"""

import logging
from typing import Dict, Set, Optional, List
from datetime import timedelta
import hashlib
import json

logger = logging.getLogger(__name__)


class NIDSStateManager:
    """Manages NIDS state in Redis for speed"""

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        """
        Initialize Redis state manager

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
        """
        self.host = host
        self.port = port
        self.db = db
        self.redis = None
        self.pipeline = None
        self._connect()

    def _connect(self):
        """Connect to Redis"""
        try:
            import redis

            self.redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )

            # Test connection
            self.redis.ping()

            self.pipeline = self.redis.pipeline()

            logger.info(f"✅ Connected to Redis at {self.host}:{self.port}")

        except ImportError:
            logger.error("redis not installed. Install with: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("Redis functionality will be disabled")
            self.redis = None

    def track_connection(self, src_ip: str, dst_ip: str, dst_port: int) -> int:
        """
        Track connection for pattern detection

        Args:
            src_ip: Source IP address
            dst_ip: Destination IP address
            dst_port: Destination port

        Returns:
            Connection count in current window
        """
        if not self.redis:
            return 0

        try:
            key = f"conn:{src_ip}:{dst_ip}:{dst_port}"

            # Increment counter with expiry
            count = self.redis.incr(key)
            self.redis.expire(key, 60)  # 60 second window

            return int(count)

        except Exception as e:
            logger.debug(f"Error tracking connection: {e}")
            return 0

    def detect_port_scan(self, src_ip: str, threshold: int = 20) -> bool:
        """
        Detect port scanning using Redis

        Args:
            src_ip: Source IP address
            threshold: Number of unique ports to trigger alert

        Returns:
            True if port scan detected
        """
        if not self.redis:
            return False

        try:
            # Get all ports accessed by this IP in last minute
            pattern = f"conn:{src_ip}:*"
            keys = self.redis.keys(pattern)

            unique_ports = set()
            for key in keys:
                parts = key.split(':')
                if len(parts) >= 4:
                    unique_ports.add(parts[3])

            # Port scan = >threshold unique ports in 60 seconds
            return len(unique_ports) > threshold

        except Exception as e:
            logger.debug(f"Error detecting port scan: {e}")
            return False

    def detect_syn_flood(self, dst_ip: str, threshold: int = 100) -> bool:
        """
        Detect SYN flood attack

        Args:
            dst_ip: Destination IP address
            threshold: SYN packets per second to trigger alert

        Returns:
            True if SYN flood detected
        """
        if not self.redis:
            return False

        try:
            key = f"syn:{dst_ip}"
            count = self.redis.get(key)

            return count and int(count) > threshold

        except Exception as e:
            logger.debug(f"Error detecting SYN flood: {e}")
            return False

    def track_syn_packet(self, dst_ip: str):
        """
        Track SYN packet for flood detection

        Args:
            dst_ip: Destination IP address
        """
        if not self.redis:
            return

        try:
            key = f"syn:{dst_ip}"
            self.redis.incr(key)
            self.redis.expire(key, 1)  # 1 second window

        except Exception as e:
            logger.debug(f"Error tracking SYN: {e}")

    def cache_ml_prediction(self, packet_hash: str, prediction: int):
        """
        Cache ML predictions for repeated patterns

        Args:
            packet_hash: Hash of packet features
            prediction: Model prediction (0=normal, 1=attack)
        """
        if not self.redis:
            return

        try:
            key = f"ml_cache:{packet_hash}"
            self.redis.setex(key, timedelta(hours=1), prediction)

        except Exception as e:
            logger.debug(f"Error caching prediction: {e}")

    def get_cached_prediction(self, packet_hash: str) -> Optional[int]:
        """
        Get cached prediction

        Args:
            packet_hash: Hash of packet features

        Returns:
            Cached prediction or None
        """
        if not self.redis:
            return None

        try:
            result = self.redis.get(f"ml_cache:{packet_hash}")
            return int(result) if result else None

        except Exception as e:
            logger.debug(f"Error getting cached prediction: {e}")
            return None

    def compute_packet_hash(self, packet: Dict) -> str:
        """
        Compute hash of packet for caching

        Args:
            packet: Packet dictionary

        Returns:
            SHA256 hash of packet features
        """
        # Extract relevant features for hashing
        features = f"{packet.get('src_ip')}:{packet.get('dst_ip')}:" \
                  f"{packet.get('protocol')}:{packet.get('dst_port')}"

        return hashlib.sha256(features.encode()).hexdigest()

    def update_statistics(self, stat_name: str, value: int = 1):
        """
        Update real-time statistics

        Args:
            stat_name: Name of statistic
            value: Value to add
        """
        if not self.redis:
            return

        try:
            self.redis.hincrby('nids:stats', stat_name, value)

        except Exception as e:
            logger.debug(f"Error updating statistics: {e}")

    def get_statistics(self) -> Dict:
        """
        Get all statistics

        Returns:
            Dictionary of statistics
        """
        if not self.redis:
            return {}

        try:
            return self.redis.hgetall('nids:stats')

        except Exception as e:
            logger.debug(f"Error getting statistics: {e}")
            return {}

    def reset_statistics(self):
        """Reset all statistics"""
        if not self.redis:
            return

        try:
            self.redis.delete('nids:stats')
            logger.info("Statistics reset")

        except Exception as e:
            logger.debug(f"Error resetting statistics: {e}")

    def add_to_blacklist(self, ip: str, ttl: int = 3600):
        """
        Add IP to blacklist

        Args:
            ip: IP address to blacklist
            ttl: Time to live in seconds
        """
        if not self.redis:
            return

        try:
            self.redis.setex(f"blacklist:{ip}", ttl, 1)
            logger.warning(f"Added {ip} to blacklist for {ttl} seconds")

        except Exception as e:
            logger.debug(f"Error adding to blacklist: {e}")

    def is_blacklisted(self, ip: str) -> bool:
        """
        Check if IP is blacklisted

        Args:
            ip: IP address to check

        Returns:
            True if blacklisted
        """
        if not self.redis:
            return False

        try:
            return self.redis.exists(f"blacklist:{ip}") > 0

        except Exception as e:
            logger.debug(f"Error checking blacklist: {e}")
            return False

    def get_blacklist(self) -> List[str]:
        """
        Get all blacklisted IPs

        Returns:
            List of blacklisted IP addresses
        """
        if not self.redis:
            return []

        try:
            keys = self.redis.keys("blacklist:*")
            return [key.replace("blacklist:", "") for key in keys]

        except Exception as e:
            logger.debug(f"Error getting blacklist: {e}")
            return []

    def close(self):
        """Close Redis connection"""
        if self.redis:
            self.redis.close()
            logger.info("Redis connection closed")


# Example usage
def example_usage():
    """Example of using Redis state manager"""

    # Initialize state manager
    state = NIDSStateManager()

    # Track connections
    print("\n=== Connection Tracking ===")
    for i in range(5):
        count = state.track_connection('192.168.1.100', '10.0.0.5', 80)
        print(f"Connection count: {count}")

    # Detect port scan
    print("\n=== Port Scan Detection ===")
    for port in range(1, 25):
        state.track_connection('192.168.1.100', '10.0.0.5', port)

    if state.detect_port_scan('192.168.1.100'):
        print("🚨 PORT SCAN DETECTED!")

    # ML prediction caching
    print("\n=== ML Prediction Caching ===")
    packet = {
        'src_ip': '192.168.1.100',
        'dst_ip': '10.0.0.5',
        'protocol': 'TCP',
        'dst_port': 80
    }

    packet_hash = state.compute_packet_hash(packet)
    state.cache_ml_prediction(packet_hash, 1)

    cached = state.get_cached_prediction(packet_hash)
    print(f"Cached prediction: {cached}")

    # Statistics
    print("\n=== Statistics ===")
    state.update_statistics('packets_processed', 100)
    state.update_statistics('attacks_detected', 5)

    stats = state.get_statistics()
    print(f"Stats: {stats}")

    # Blacklist
    print("\n=== Blacklist ===")
    state.add_to_blacklist('192.168.1.100', ttl=10)

    if state.is_blacklisted('192.168.1.100'):
        print("IP is blacklisted")

    print(f"Blacklist: {state.get_blacklist()}")

    # Cleanup
    state.close()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Redis State Manager Example\n")
    example_usage()
    print("\nExample complete!")
