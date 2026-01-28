"""
Automated Threat Response System
Implements automated blocking, isolation, and notification
"""

import subprocess
import requests
import logging
from typing import Dict, Set, Optional
from datetime import datetime
import platform

logger = logging.getLogger(__name__)


class AutomatedResponse:
    """Automatically respond to detected threats"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize automated response system

        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self.blocked_ips = set()
        self.rate_limited_ips = set()
        self.response_log = []
        self.os_type = platform.system()

        logger.info(f"✅ Automated response initialized on {self.os_type}")

    def handle_alert(self, alert: Dict):
        """
        Handle alert based on severity

        Args:
            alert: Alert dictionary with attack information
        """
        severity = alert.get('severity', 'MEDIUM')
        attack_type = alert.get('attack_type', 'UNKNOWN')
        src_ip = alert.get('source_ip') or alert.get('packet', {}).get('src_ip')

        logger.info(f"Handling {severity} alert: {attack_type} from {src_ip}")

        # Log response
        self._log_response(alert, severity)

        # Handle based on severity
        if severity == 'CRITICAL':
            self._handle_critical(src_ip, attack_type, alert)

        elif severity == 'HIGH':
            self._handle_high(src_ip, attack_type, alert)

        elif severity == 'MEDIUM':
            self._handle_medium(src_ip, attack_type, alert)

        else:  # LOW
            self._handle_low(src_ip, attack_type, alert)

    def _handle_critical(self, ip: str, attack_type: str, alert: Dict):
        """Handle critical severity alerts"""
        logger.warning(f"🚨 CRITICAL ALERT: {attack_type} from {ip}")

        # Block IP immediately
        self._block_ip(ip)

        # Isolate device (if configured)
        if self.config.get('auto_isolate', False):
            self._isolate_device(ip)

        # Notify SOC/PagerDuty
        self._notify_soc(alert)

        # Notify admin immediately
        self._notify_admin(alert, urgent=True)

    def _handle_high(self, ip: str, attack_type: str, alert: Dict):
        """Handle high severity alerts"""
        logger.warning(f"⚠️  HIGH ALERT: {attack_type} from {ip}")

        # Rate limit instead of full block
        self._rate_limit_ip(ip)

        # Notify admin
        self._notify_admin(alert)

        # Log to SIEM
        self._log_to_siem(alert)

    def _handle_medium(self, ip: str, attack_type: str, alert: Dict):
        """Handle medium severity alerts"""
        logger.info(f"ℹ️  MEDIUM ALERT: {attack_type} from {ip}")

        # Just log
        self._log_warning(alert)

        # Track for patterns
        self._track_suspicious_ip(ip)

    def _handle_low(self, ip: str, attack_type: str, alert: Dict):
        """Handle low severity alerts"""
        logger.debug(f"LOW ALERT: {attack_type} from {ip}")

        # Only log
        self._log_warning(alert)

    def _block_ip(self, ip: str) -> bool:
        """
        Block IP at firewall

        Args:
            ip: IP address to block

        Returns:
            True if successful
        """
        if not ip or ip in self.blocked_ips:
            return False

        try:
            if self.os_type == 'Windows':
                # Windows Firewall
                cmd = f'netsh advfirewall firewall add rule name="NIDS Block {ip}" dir=in action=block remoteip={ip}'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                if result.returncode == 0:
                    self.blocked_ips.add(ip)
                    logger.warning(f"🚫 BLOCKED IP: {ip}")
                    return True
                else:
                    logger.error(f"Failed to block IP: {result.stderr}")
                    return False

            elif self.os_type == 'Linux':
                # Linux iptables
                cmd = f'iptables -A INPUT -s {ip} -j DROP'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                if result.returncode == 0:
                    self.blocked_ips.add(ip)
                    logger.warning(f"🚫 BLOCKED IP: {ip}")
                    return True
                else:
                    logger.error(f"Failed to block IP: {result.stderr}")
                    return False
            else:
                logger.warning(f"IP blocking not supported on {self.os_type}")
                return False

        except Exception as e:
            logger.error(f"Error blocking IP {ip}: {e}")
            return False

    def _unblock_ip(self, ip: str) -> bool:
        """
        Unblock IP at firewall

        Args:
            ip: IP address to unblock

        Returns:
            True if successful
        """
        if ip not in self.blocked_ips:
            return False

        try:
            if self.os_type == 'Windows':
                cmd = f'netsh advfirewall firewall delete rule name="NIDS Block {ip}"'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                if result.returncode == 0:
                    self.blocked_ips.remove(ip)
                    logger.info(f"✅ UNBLOCKED IP: {ip}")
                    return True

            elif self.os_type == 'Linux':
                cmd = f'iptables -D INPUT -s {ip} -j DROP'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                if result.returncode == 0:
                    self.blocked_ips.remove(ip)
                    logger.info(f"✅ UNBLOCKED IP: {ip}")
                    return True

        except Exception as e:
            logger.error(f"Error unblocking IP {ip}: {e}")
            return False

    def _isolate_device(self, ip: str):
        """
        Isolate infected device on network

        Args:
            ip: IP address to isolate
        """
        # This is network-dependent and requires switch/router API
        logger.warning(f"Device isolation requested for {ip} (not implemented - requires network API)")

        # Example: Call network switch API to move device to quarantine VLAN
        # if self.config.get('network_api_url'):
        #     response = requests.post(
        #         f"{self.config['network_api_url']}/isolate",
        #         json={'ip': ip},
        #         headers={'Authorization': f"Bearer {self.config['network_api_token']}"}
        #     )

    def _rate_limit_ip(self, ip: str):
        """
        Apply rate limiting to IP

        Args:
            ip: IP address to rate limit
        """
        if ip in self.rate_limited_ips:
            return

        self.rate_limited_ips.add(ip)
        logger.warning(f"⏱️  RATE LIMITED: {ip}")

        # Implementation depends on load balancer/firewall
        # This is a placeholder

    def _notify_soc(self, alert: Dict):
        """
        Notify Security Operations Center

        Args:
            alert: Alert dictionary
        """
        # PagerDuty integration
        if self.config.get('pagerduty_key'):
            try:
                response = requests.post(
                    'https://events.pagerduty.com/v2/enqueue',
                    json={
                        'routing_key': self.config['pagerduty_key'],
                        'event_action': 'trigger',
                        'payload': {
                            'summary': f"CRITICAL: {alert.get('attack_type')} detected",
                            'severity': 'critical',
                            'source': alert.get('source_ip'),
                            'timestamp': datetime.now().isoformat(),
                            'custom_details': alert
                        }
                    },
                    timeout=5
                )

                if response.status_code == 202:
                    logger.info("✅ PagerDuty alert sent")
                else:
                    logger.error(f"PagerDuty error: {response.status_code}")

            except Exception as e:
                logger.error(f"Error sending PagerDuty alert: {e}")

    def _notify_admin(self, alert: Dict, urgent: bool = False):
        """
        Notify administrator

        Args:
            alert: Alert dictionary
            urgent: Whether this is urgent
        """
        # Send to webhook
        if self.config.get('webhook_url'):
            try:
                response = requests.post(
                    self.config['webhook_url'],
                    json={
                        'urgent': urgent,
                        'alert': alert,
                        'timestamp': datetime.now().isoformat()
                    },
                    timeout=5
                )

                if response.status_code == 200:
                    logger.info("✅ Webhook notification sent")

            except Exception as e:
                logger.error(f"Error sending webhook: {e}")

        # Could also send email, SMS, etc.

    def _log_to_siem(self, alert: Dict):
        """
        Log to SIEM (Splunk, ELK, etc.)

        Args:
            alert: Alert dictionary
        """
        if self.config.get('siem_url'):
            try:
                response = requests.post(
                    self.config['siem_url'],
                    json=alert,
                    headers={'Authorization': f"Bearer {self.config.get('siem_token', '')}"},
                    timeout=5
                )

                if response.status_code == 200:
                    logger.info("✅ SIEM log sent")

            except Exception as e:
                logger.error(f"Error sending to SIEM: {e}")

    def _log_warning(self, alert: Dict):
        """Log warning to file"""
        logger.warning(f"Security alert: {alert}")

        # Could also write to dedicated security log file

    def _track_suspicious_ip(self, ip: str):
        """Track suspicious IP for pattern analysis"""
        # Could use Redis or database to track
        logger.debug(f"Tracking suspicious IP: {ip}")

    def _log_response(self, alert: Dict, severity: str):
        """Log response action"""
        self.response_log.append({
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'attack_type': alert.get('attack_type'),
            'source_ip': alert.get('source_ip') or alert.get('packet', {}).get('src_ip')
        })

    def get_blocked_ips(self) -> Set[str]:
        """Get set of blocked IPs"""
        return self.blocked_ips.copy()

    def get_response_log(self) -> list:
        """Get response log"""
        return self.response_log.copy()


# Example usage
def example_usage():
    """Example of using automated response"""

    # Create response system with config
    config = {
        'auto_isolate': False,
        'webhook_url': 'http://localhost:5000/webhook',
        # 'pagerduty_key': 'YOUR_KEY_HERE',
        # 'siem_url': 'https://your-siem.com/api/events',
        # 'siem_token': 'YOUR_TOKEN_HERE'
    }

    response_system = AutomatedResponse(config)

    # Simulate alerts
    alerts = [
        {
            'severity': 'CRITICAL',
            'attack_type': 'DDoS',
            'source_ip': '192.168.1.100',
            'packet': {'src_ip': '192.168.1.100', 'dst_ip': '10.0.0.5'}
        },
        {
            'severity': 'HIGH',
            'attack_type': 'Port Scan',
            'source_ip': '192.168.1.101',
            'packet': {'src_ip': '192.168.1.101', 'dst_ip': '10.0.0.5'}
        },
        {
            'severity': 'MEDIUM',
            'attack_type': 'Suspicious Activity',
            'source_ip': '192.168.1.102',
            'packet': {'src_ip': '192.168.1.102', 'dst_ip': '10.0.0.5'}
        }
    ]

    print("\nProcessing security alerts...\n")

    for alert in alerts:
        response_system.handle_alert(alert)
        print()

    # Show blocked IPs
    print(f"\nBlocked IPs: {response_system.get_blocked_ips()}")
    print(f"Response log entries: {len(response_system.get_response_log())}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=== Automated Response System ===\n")
    example_usage()
