"""
Real-Time Network Packet Capture Module
Implements live network traffic capture with threading and queuing
"""

import threading
import queue
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RealTimePacketCapture:
    """
    Real-time packet capture using Scapy with threading support
    Captures live network traffic and processes packets asynchronously
    """

    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize real-time packet capture

        Args:
            max_queue_size: Maximum number of packets to queue
        """
        self.packet_queue = queue.Queue(maxsize=max_queue_size)
        self.is_capturing = False
        self.capture_thread = None
        self.packets_captured = 0
        self.packets_dropped = 0

        # Try to import Scapy
        try:
            from scapy.all import sniff, IP, TCP, UDP, ICMP, Ether
            self.sniff = sniff
            self.IP = IP
            self.TCP = TCP
            self.UDP = UDP
            self.ICMP = ICMP
            self.Ether = Ether
            self.scapy_available = True
            logger.info("✅ Scapy loaded - Real packet capture enabled")
        except ImportError:
            self.scapy_available = False
            logger.warning("⚠️  Scapy not installed - Real capture unavailable")
            logger.warning("Install: pip install scapy")

    def packet_handler(self, packet):
        """
        Handle each captured packet

        Args:
            packet: Scapy packet object
        """
        if not packet.haslayer(self.IP):
            return

        try:
            packet_data = self._parse_packet(packet)
            if packet_data:
                # Try to add to queue
                try:
                    self.packet_queue.put_nowait(packet_data)
                    self.packets_captured += 1
                except queue.Full:
                    self.packets_dropped += 1
                    logger.warning(f"Queue full! Dropped packet (Total dropped: {self.packets_dropped})")
        except Exception as e:
            logger.error(f"Error handling packet: {e}")

    def _parse_packet(self, pkt) -> Optional[Dict]:
        """
        Parse Scapy packet into standardized dictionary format

        Args:
            pkt: Scapy packet object

        Returns:
            Dictionary containing packet information
        """
        try:
            packet_data = {
                'timestamp': datetime.now(),
                'src_ip': pkt[self.IP].src,
                'dst_ip': pkt[self.IP].dst,
                'packet_size': len(pkt),
                'protocol': 'UNKNOWN',
                'src_port': 0,
                'dst_port': 0,
                'flags': '',
                'ttl': pkt[self.IP].ttl if hasattr(pkt[self.IP], 'ttl') else 0,
                'window_size': 0,
                'urgent_pointer': 0
            }

            # Protocol-specific parsing
            if pkt.haslayer(self.TCP):
                packet_data.update({
                    'protocol': 'TCP',
                    'src_port': pkt[self.TCP].sport,
                    'dst_port': pkt[self.TCP].dport,
                    'flags': str(pkt[self.TCP].flags),
                    'window_size': pkt[self.TCP].window,
                    'urgent_pointer': pkt[self.TCP].urgptr if hasattr(pkt[self.TCP], 'urgptr') else 0,
                    'seq_number': pkt[self.TCP].seq,
                    'ack_number': pkt[self.TCP].ack,
                })
            elif pkt.haslayer(self.UDP):
                packet_data.update({
                    'protocol': 'UDP',
                    'src_port': pkt[self.UDP].sport,
                    'dst_port': pkt[self.UDP].dport,
                    'length': pkt[self.UDP].len,
                })
            elif pkt.haslayer(self.ICMP):
                packet_data.update({
                    'protocol': 'ICMP',
                    'icmp_type': pkt[self.ICMP].type,
                    'icmp_code': pkt[self.ICMP].code,
                })

            return packet_data

        except Exception as e:
            logger.error(f"Error parsing packet: {e}")
            return None

    def start_capture(self, interface: Optional[str] = None,
                     packet_count: Optional[int] = None,
                     filter_bpf: Optional[str] = None):
        """
        Start capturing packets in a background thread

        Args:
            interface: Network interface to capture on (None = all interfaces)
            packet_count: Maximum packets to capture (None = unlimited)
            filter_bpf: BPF filter string (e.g., "tcp port 80")
        """
        if not self.scapy_available:
            logger.error("Scapy not available. Cannot start real capture.")
            return False

        if self.is_capturing:
            logger.warning("Capture already in progress")
            return False

        self.is_capturing = True
        logger.info(f"🚀 Starting packet capture...")
        logger.info(f"   Interface: {interface or 'all'}")
        logger.info(f"   Filter: {filter_bpf or 'none'}")
        logger.info(f"   Count: {packet_count or 'unlimited'}")

        def capture_worker():
            """Worker thread for packet capture"""
            try:
                self.sniff(
                    iface=interface,
                    prn=self.packet_handler,
                    store=False,
                    count=packet_count,
                    filter=filter_bpf,
                    stop_filter=lambda x: not self.is_capturing
                )
            except Exception as e:
                logger.error(f"Capture error: {e}")
            finally:
                self.is_capturing = False
                logger.info("Packet capture stopped")

        self.capture_thread = threading.Thread(target=capture_worker, daemon=True)
        self.capture_thread.start()
        return True

    def stop_capture(self):
        """Stop packet capture"""
        if self.is_capturing:
            logger.info("Stopping packet capture...")
            self.is_capturing = False
            if self.capture_thread:
                self.capture_thread.join(timeout=5)
            logger.info(f"Capture stopped. Packets captured: {self.packets_captured}")

    def get_packet(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get next packet from queue

        Args:
            timeout: Maximum time to wait for packet

        Returns:
            Packet dictionary or None if timeout
        """
        try:
            return self.packet_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self) -> Dict:
        """Get capture statistics"""
        return {
            'packets_captured': self.packets_captured,
            'packets_dropped': self.packets_dropped,
            'queue_size': self.packet_queue.qsize(),
            'is_capturing': self.is_capturing
        }


class StreamingDetector:
    """
    Sliding window analyzer for real-time pattern detection
    Detects attack patterns in streaming network traffic
    """

    def __init__(self, window_size: int = 100, window_time: int = 60):
        """
        Initialize streaming detector

        Args:
            window_size: Maximum number of packets in window
            window_time: Time window in seconds
        """
        from collections import deque
        self.packet_window = deque(maxlen=window_size)
        self.window_time = window_time
        self.window_size = window_size

        # Attack detection thresholds
        self.port_scan_threshold = 20
        self.ddos_threshold = 50
        self.syn_flood_threshold = 30

    def add_packet(self, packet: Dict):
        """
        Add packet to sliding window

        Args:
            packet: Packet dictionary
        """
        packet['capture_time'] = time.time()
        self.packet_window.append(packet)
        self._cleanup_old_packets()

    def _cleanup_old_packets(self):
        """Remove packets older than window_time"""
        current_time = time.time()
        while (self.packet_window and
               current_time - self.packet_window[0]['capture_time'] > self.window_time):
            self.packet_window.popleft()

    def analyze_window(self) -> Optional[Dict]:
        """
        Analyze current window for attack patterns

        Returns:
            Dictionary with attack info if detected, None otherwise
        """
        if len(self.packet_window) < 10:
            return None

        # Port scan detection
        unique_dst_ports = set()
        src_ips = []
        syn_packets = 0
        protocol_counts = {'TCP': 0, 'UDP': 0, 'ICMP': 0}

        for pkt in self.packet_window:
            unique_dst_ports.add(pkt.get('dst_port', 0))
            src_ips.append(pkt.get('src_ip', ''))

            if pkt.get('flags') and 'S' in pkt.get('flags', ''):
                syn_packets += 1

            protocol = pkt.get('protocol', 'UNKNOWN')
            if protocol in protocol_counts:
                protocol_counts[protocol] += 1

        # Port scan detection
        if len(unique_dst_ports) > self.port_scan_threshold:
            return {
                'type': 'Port Scan',
                'severity': 'HIGH',
                'details': f'{len(unique_dst_ports)} unique ports accessed',
                'confidence': 0.92
            }

        # DDoS detection (many packets from same source)
        from collections import Counter
        src_counter = Counter(src_ips)
        most_common_src, count = src_counter.most_common(1)[0]

        if count > self.ddos_threshold:
            return {
                'type': 'DDoS Attack',
                'severity': 'CRITICAL',
                'details': f'{count} packets from {most_common_src}',
                'confidence': 0.95
            }

        # SYN flood detection
        if syn_packets > self.syn_flood_threshold:
            return {
                'type': 'SYN Flood',
                'severity': 'HIGH',
                'details': f'{syn_packets} SYN packets in window',
                'confidence': 0.90
            }

        # ICMP flood detection
        if protocol_counts['ICMP'] > 40:
            return {
                'type': 'ICMP Flood',
                'severity': 'HIGH',
                'details': f'{protocol_counts["ICMP"]} ICMP packets',
                'confidence': 0.88
            }

        return None

    def get_window_stats(self) -> Dict:
        """Get statistics about current window"""
        if not self.packet_window:
            return {}

        protocols = [p.get('protocol', 'UNKNOWN') for p in self.packet_window]
        from collections import Counter
        protocol_dist = Counter(protocols)

        return {
            'window_size': len(self.packet_window),
            'unique_src_ips': len(set(p.get('src_ip', '') for p in self.packet_window)),
            'unique_dst_ips': len(set(p.get('dst_ip', '') for p in self.packet_window)),
            'unique_dst_ports': len(set(p.get('dst_port', 0) for p in self.packet_window)),
            'protocol_distribution': dict(protocol_dist)
        }


# Example usage
if __name__ == '__main__':
    print("Real-Time Packet Capture Module")
    print("=" * 70)

    # Initialize capture
    capturer = RealTimePacketCapture()
    detector = StreamingDetector(window_size=100, window_time=60)

    if capturer.scapy_available:
        print("\n[*] Starting packet capture for 30 seconds...")
        print("[*] Press Ctrl+C to stop early\n")

        # Start capture
        capturer.start_capture(filter_bpf="ip", packet_count=100)

        try:
            # Process packets for 30 seconds
            start_time = time.time()
            while time.time() - start_time < 30:
                packet = capturer.get_packet(timeout=1.0)

                if packet:
                    # Add to detector
                    detector.add_packet(packet)

                    # Analyze for attacks
                    attack = detector.analyze_window()
                    if attack:
                        print(f"\n🚨 ALERT: {attack['type']} - {attack['severity']}")
                        print(f"   Details: {attack['details']}")
                        print(f"   Confidence: {attack['confidence']*100:.1f}%\n")

                    # Show stats every 10 packets
                    if capturer.packets_captured % 10 == 0:
                        stats = capturer.get_stats()
                        print(f"Captured: {stats['packets_captured']} | "
                              f"Queue: {stats['queue_size']} | "
                              f"Dropped: {stats['packets_dropped']}")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            # Stop capture
            capturer.stop_capture()

            # Final stats
            print("\n" + "=" * 70)
            print("CAPTURE SUMMARY")
            print("=" * 70)
            stats = capturer.get_stats()
            window_stats = detector.get_window_stats()

            print(f"Total packets captured: {stats['packets_captured']}")
            print(f"Packets dropped: {stats['packets_dropped']}")
            print(f"\nWindow statistics:")
            for key, value in window_stats.items():
                print(f"  {key}: {value}")
    else:
        print("\n❌ Scapy not available. Install with: pip install scapy")
