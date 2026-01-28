"""
High-Speed Packet Capture Module (v3.0)
Implements ultra-fast packet capture using PyShark and LibPCAP
"""

import asyncio
from typing import Dict, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class HighSpeedCapture:
    """Ultra-fast packet capture using PyShark"""

    def __init__(self, interface: str = 'eth0'):
        """
        Initialize high-speed capture

        Args:
            interface: Network interface to capture from
        """
        self.interface = interface
        self.capture = None
        self.packet_callback = None
        self.is_running = False

    async def start_async_capture(self, callback: Callable):
        """
        Asynchronous packet capture - non-blocking

        Args:
            callback: Async function to call for each packet
        """
        try:
            import pyshark

            self.packet_callback = callback
            self.is_running = True

            logger.info(f"Starting async capture on {self.interface}")

            self.capture = pyshark.LiveCapture(
                interface=self.interface,
                bpf_filter='tcp or udp or icmp'
            )

            async for packet in self.capture.sniff_continuously():
                if not self.is_running:
                    break

                if self.packet_callback:
                    parsed = self.parse_packet(packet)
                    if parsed:
                        await self.packet_callback(parsed)

        except ImportError:
            logger.error("PyShark not installed. Install with: pip install pyshark")
            raise
        except Exception as e:
            logger.error(f"Capture error: {e}")
            raise

    def parse_packet(self, packet) -> Optional[Dict]:
        """
        Parse PyShark packet to standard format

        Args:
            packet: PyShark packet object

        Returns:
            Dictionary with packet features or None if parsing fails
        """
        try:
            return {
                'timestamp': packet.sniff_time,
                'src_ip': packet.ip.src if hasattr(packet, 'ip') else None,
                'dst_ip': packet.ip.dst if hasattr(packet, 'ip') else None,
                'protocol': packet.transport_layer if hasattr(packet, 'transport_layer') else 'OTHER',
                'packet_size': int(packet.length),
                'src_port': int(packet[packet.transport_layer].srcport) if hasattr(packet, packet.transport_layer) and hasattr(packet[packet.transport_layer], 'srcport') else 0,
                'dst_port': int(packet[packet.transport_layer].dstport) if hasattr(packet, packet.transport_layer) and hasattr(packet[packet.transport_layer], 'dstport') else 0,
            }
        except Exception as e:
            logger.debug(f"Packet parsing error: {e}")
            return None

    def stop(self):
        """Stop packet capture"""
        self.is_running = False
        if self.capture:
            self.capture.close()
        logger.info("Capture stopped")


class UltraFastCapture:
    """Maximum performance using libpcap direct bindings"""

    def __init__(self, interface: str = 'eth0'):
        """
        Initialize ultra-fast capture

        Args:
            interface: Network interface to capture from
        """
        self.interface = interface
        self.decoder = None
        self.is_running = False

    def start_capture(self, callback: Callable, packet_limit: int = 0):
        """
        Capture packets at maximum speed

        Args:
            callback: Function to call for each packet
            packet_limit: Maximum packets to capture (0 = unlimited)
        """
        try:
            import pcapy
            from impacket.ImpactDecoder import EthDecoder

            self.decoder = EthDecoder()
            self.is_running = True

            logger.info(f"Starting ultra-fast capture on {self.interface}")

            # Open interface
            cap = pcapy.open_live(self.interface, 65536, 1, 0)

            # Set filter for efficiency
            cap.setfilter('tcp or udp or icmp')

            # Capture loop
            cap.loop(packet_limit, lambda header, data:
                     self._process_packet(header, data, callback))

        except ImportError:
            logger.error("Pcapy/Impacket not installed. Install with: pip install pcapy impacket")
            raise
        except Exception as e:
            logger.error(f"Capture error: {e}")
            raise

    def _process_packet(self, header, data, callback):
        """
        Process packet with minimal overhead

        Args:
            header: Packet header
            data: Packet data
            callback: Function to call with extracted features
        """
        if not self.is_running:
            return

        try:
            eth = self.decoder.decode(data)
            features = self._extract_features(eth, header)
            if features:
                callback(features)
        except Exception as e:
            logger.debug(f"Packet processing error: {e}")

    def _extract_features(self, packet, header) -> Optional[Dict]:
        """
        Fast feature extraction

        Args:
            packet: Decoded packet
            header: Packet header

        Returns:
            Dictionary with packet features or None if extraction fails
        """
        try:
            from datetime import datetime

            features = {
                'timestamp': datetime.fromtimestamp(header.getts()[0]),
                'packet_size': header.getlen(),
                'src_ip': None,
                'dst_ip': None,
                'protocol': 'OTHER',
                'src_port': 0,
                'dst_port': 0
            }

            # Extract IP layer
            ip_packet = packet.child()
            if ip_packet:
                features['src_ip'] = ip_packet.get_ip_src()
                features['dst_ip'] = ip_packet.get_ip_dst()
                features['protocol'] = ip_packet.get_ip_p()

                # Extract transport layer
                transport = ip_packet.child()
                if transport:
                    if hasattr(transport, 'get_th_sport'):
                        features['src_port'] = transport.get_th_sport()
                        features['dst_port'] = transport.get_th_dport()
                        features['protocol'] = 'TCP'
                    elif hasattr(transport, 'get_uh_sport'):
                        features['src_port'] = transport.get_uh_sport()
                        features['dst_port'] = transport.get_uh_dport()
                        features['protocol'] = 'UDP'

            return features
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None

    def stop(self):
        """Stop packet capture"""
        self.is_running = False
        logger.info("Capture stopped")


# Example usage
async def example_async_capture():
    """Example of using async high-speed capture"""

    async def packet_handler(packet):
        print(f"Packet from {packet['src_ip']} to {packet['dst_ip']}")

    capture = HighSpeedCapture(interface='eth0')
    await capture.start_async_capture(packet_handler)


def example_sync_capture():
    """Example of using synchronous ultra-fast capture"""

    def packet_handler(packet):
        print(f"Packet from {packet['src_ip']} to {packet['dst_ip']}")

    capture = UltraFastCapture(interface='eth0')
    capture.start_capture(packet_handler, packet_limit=100)


if __name__ == "__main__":
    # Run async example
    print("Running high-speed async capture...")
    asyncio.run(example_async_capture())
