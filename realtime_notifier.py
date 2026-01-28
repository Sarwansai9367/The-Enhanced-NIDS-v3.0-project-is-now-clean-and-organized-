"""
Real-Time Alert Notification System
Implements multiple notification channels: Email, SMS, Webhook, Console
"""

import smtplib
import requests
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertNotifier:
    """
    Multi-channel alert notification system
    Supports: Email, Webhook, Console, File logging
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize alert notifier

        Args:
            config_file: Path to configuration JSON file
        """
        self.config = self._load_config(config_file)
        self.notification_log = []
        logger.info("✅ Alert Notifier initialized")

    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from_email': '',
                'to_emails': [],
                'use_tls': True
            },
            'webhook': {
                'enabled': False,
                'url': '',
                'headers': {},
                'method': 'POST'
            },
            'console': {
                'enabled': True,
                'color': True
            },
            'file': {
                'enabled': True,
                'path': 'alerts.log'
            },
            'slack': {
                'enabled': False,
                'webhook_url': ''
            },
            'telegram': {
                'enabled': False,
                'bot_token': '',
                'chat_id': ''
            }
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    logger.info(f"Loaded config from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")

        return default_config

    def send_alert(self, alert_data: Dict, priority: str = 'MEDIUM'):
        """
        Send alert through all enabled channels

        Args:
            alert_data: Alert information dictionary
            priority: Alert priority (LOW, MEDIUM, HIGH, CRITICAL)
        """
        # Format alert message
        message = self._format_alert_message(alert_data)

        # Log notification
        self.notification_log.append({
            'timestamp': datetime.now(),
            'alert_type': alert_data.get('attack_type', 'Unknown'),
            'priority': priority,
            'channels': []
        })

        # Send through enabled channels
        if self.config['console']['enabled']:
            self._send_console_alert(alert_data, priority)
            self.notification_log[-1]['channels'].append('console')

        if self.config['file']['enabled']:
            self._send_file_alert(message, priority)
            self.notification_log[-1]['channels'].append('file')

        if self.config['email']['enabled']:
            try:
                self._send_email_alert(alert_data, message, priority)
                self.notification_log[-1]['channels'].append('email')
            except Exception as e:
                logger.error(f"Email notification failed: {e}")

        if self.config['webhook']['enabled']:
            try:
                self._send_webhook_alert(alert_data, priority)
                self.notification_log[-1]['channels'].append('webhook')
            except Exception as e:
                logger.error(f"Webhook notification failed: {e}")

        if self.config['slack']['enabled']:
            try:
                self._send_slack_alert(alert_data, priority)
                self.notification_log[-1]['channels'].append('slack')
            except Exception as e:
                logger.error(f"Slack notification failed: {e}")

        if self.config['telegram']['enabled']:
            try:
                self._send_telegram_alert(alert_data, priority)
                self.notification_log[-1]['channels'].append('telegram')
            except Exception as e:
                logger.error(f"Telegram notification failed: {e}")

    def _format_alert_message(self, alert_data: Dict) -> str:
        """Format alert as text message"""
        return f"""
🚨 NETWORK INTRUSION DETECTION ALERT

Timestamp:        {alert_data.get('timestamp', datetime.now())}
Severity:         {alert_data.get('severity', 'UNKNOWN')}
Attack Type:      {alert_data.get('attack_type', 'Unknown')}
Source IP:        {alert_data.get('source_ip', 'Unknown')}
Destination IP:   {alert_data.get('destination_ip', 'Unknown')}
Protocol:         {alert_data.get('protocol', 'Unknown')}
Detection Method: {alert_data.get('detection_method', 'Unknown')}
Confidence:       {alert_data.get('confidence', 0)*100:.1f}%

Description:      {alert_data.get('description', 'Intrusion detected')}

RECOMMENDED ACTIONS:
1. Investigate source IP immediately
2. Check firewall logs
3. Review recent network activity
4. Consider blocking source if confirmed malicious
"""

    def _send_console_alert(self, alert_data: Dict, priority: str):
        """Display alert in console with colors"""
        severity = alert_data.get('severity', 'MEDIUM')

        # Color codes
        if self.config['console']['color']:
            colors = {
                'CRITICAL': '\033[91m',  # Red
                'HIGH': '\033[93m',      # Yellow
                'MEDIUM': '\033[94m',    # Blue
                'LOW': '\033[92m',       # Green
                'RESET': '\033[0m'
            }
            color = colors.get(severity, colors['MEDIUM'])
            reset = colors['RESET']
        else:
            color = reset = ''

        print(f"\n{color}{'='*70}")
        print(f"🚨 SECURITY ALERT - {severity}")
        print(f"{'='*70}{reset}")
        print(f"{color}Time:          {alert_data.get('timestamp', datetime.now())}{reset}")
        print(f"{color}Attack Type:   {alert_data.get('attack_type', 'Unknown')}{reset}")
        print(f"{color}Source IP:     {alert_data.get('source_ip', 'Unknown')}{reset}")
        print(f"{color}Dest IP:       {alert_data.get('destination_ip', 'Unknown')}{reset}")
        print(f"{color}Protocol:      {alert_data.get('protocol', 'Unknown')}{reset}")
        print(f"{color}Detection:     {alert_data.get('detection_method', 'Unknown')}{reset}")
        print(f"{color}Confidence:    {alert_data.get('confidence', 0)*100:.1f}%{reset}")
        print(f"{color}{'='*70}{reset}\n")

    def _send_file_alert(self, message: str, priority: str):
        """Write alert to log file"""
        try:
            log_file = self.config['file']['path']
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"Priority: {priority}\n")
                f.write(message)
                f.write(f"\n{'='*70}\n")
        except Exception as e:
            logger.error(f"File logging error: {e}")

    def _send_email_alert(self, alert_data: Dict, message: str, priority: str):
        """Send alert via email"""
        if not self.config['email']['username'] or not self.config['email']['password']:
            logger.warning("Email credentials not configured")
            return

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['from_email'] or self.config['email']['username']
            msg['To'] = ', '.join(self.config['email']['to_emails'])
            msg['Subject'] = f"[NIDS {priority}] {alert_data.get('attack_type', 'Security Alert')}"

            # HTML body
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #d9534f;">🚨 Network Intrusion Alert</h2>
                <table style="border-collapse: collapse; width: 100%;">
                    <tr style="background-color: #f2f2f2;">
                        <td style="padding: 8px; border: 1px solid #ddd;"><strong>Severity</strong></td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{alert_data.get('severity', 'UNKNOWN')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;"><strong>Attack Type</strong></td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{alert_data.get('attack_type', 'Unknown')}</td>
                    </tr>
                    <tr style="background-color: #f2f2f2;">
                        <td style="padding: 8px; border: 1px solid #ddd;"><strong>Source IP</strong></td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{alert_data.get('source_ip', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;"><strong>Destination IP</strong></td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{alert_data.get('destination_ip', 'Unknown')}</td>
                    </tr>
                    <tr style="background-color: #f2f2f2;">
                        <td style="padding: 8px; border: 1px solid #ddd;"><strong>Protocol</strong></td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{alert_data.get('protocol', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;"><strong>Confidence</strong></td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{alert_data.get('confidence', 0)*100:.1f}%</td>
                    </tr>
                    <tr style="background-color: #f2f2f2;">
                        <td style="padding: 8px; border: 1px solid #ddd;"><strong>Time</strong></td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{alert_data.get('timestamp', datetime.now())}</td>
                    </tr>
                </table>
                <p><strong>Recommended Action:</strong> Investigate immediately and consider blocking source IP.</p>
            </body>
            </html>
            """

            msg.attach(MIMEText(html_body, 'html'))

            # Send email
            server = smtplib.SMTP(
                self.config['email']['smtp_server'],
                self.config['email']['smtp_port']
            )

            if self.config['email']['use_tls']:
                server.starttls()

            server.login(
                self.config['email']['username'],
                self.config['email']['password']
            )

            server.send_message(msg)
            server.quit()

            logger.info(f"Email alert sent to {len(self.config['email']['to_emails'])} recipients")

        except Exception as e:
            logger.error(f"Email send error: {e}")
            raise

    def _send_webhook_alert(self, alert_data: Dict, priority: str):
        """Send alert via webhook"""
        if not self.config['webhook']['url']:
            return

        payload = {
            'priority': priority,
            'timestamp': str(alert_data.get('timestamp', datetime.now())),
            'severity': alert_data.get('severity', 'UNKNOWN'),
            'attack_type': alert_data.get('attack_type', 'Unknown'),
            'source_ip': alert_data.get('source_ip', 'Unknown'),
            'destination_ip': alert_data.get('destination_ip', 'Unknown'),
            'protocol': alert_data.get('protocol', 'Unknown'),
            'confidence': alert_data.get('confidence', 0),
            'detection_method': alert_data.get('detection_method', 'Unknown')
        }

        headers = self.config['webhook'].get('headers', {'Content-Type': 'application/json'})

        response = requests.post(
            self.config['webhook']['url'],
            json=payload,
            headers=headers,
            timeout=5
        )

        if response.status_code == 200:
            logger.info(f"Webhook alert sent successfully")
        else:
            logger.error(f"Webhook failed: {response.status_code}")

    def _send_slack_alert(self, alert_data: Dict, priority: str):
        """Send alert to Slack"""
        if not self.config['slack']['webhook_url']:
            return

        severity_emoji = {
            'CRITICAL': '🔴',
            'HIGH': '🟠',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }

        emoji = severity_emoji.get(alert_data.get('severity', 'MEDIUM'), '⚠️')

        payload = {
            'text': f"{emoji} *Network Intrusion Alert*",
            'blocks': [
                {
                    'type': 'header',
                    'text': {
                        'type': 'plain_text',
                        'text': f"{emoji} Network Intrusion Detected"
                    }
                },
                {
                    'type': 'section',
                    'fields': [
                        {'type': 'mrkdwn', 'text': f"*Severity:*\n{alert_data.get('severity', 'UNKNOWN')}"},
                        {'type': 'mrkdwn', 'text': f"*Attack Type:*\n{alert_data.get('attack_type', 'Unknown')}"},
                        {'type': 'mrkdwn', 'text': f"*Source IP:*\n{alert_data.get('source_ip', 'Unknown')}"},
                        {'type': 'mrkdwn', 'text': f"*Dest IP:*\n{alert_data.get('destination_ip', 'Unknown')}"},
                        {'type': 'mrkdwn', 'text': f"*Protocol:*\n{alert_data.get('protocol', 'Unknown')}"},
                        {'type': 'mrkdwn', 'text': f"*Confidence:*\n{alert_data.get('confidence', 0)*100:.1f}%"}
                    ]
                }
            ]
        }

        response = requests.post(
            self.config['slack']['webhook_url'],
            json=payload,
            timeout=5
        )

        if response.status_code == 200:
            logger.info("Slack alert sent")

    def _send_telegram_alert(self, alert_data: Dict, priority: str):
        """Send alert via Telegram"""
        if not self.config['telegram']['bot_token'] or not self.config['telegram']['chat_id']:
            return

        message = f"""
🚨 *NETWORK INTRUSION ALERT*

*Severity:* {alert_data.get('severity', 'UNKNOWN')}
*Attack Type:* {alert_data.get('attack_type', 'Unknown')}
*Source IP:* {alert_data.get('source_ip', 'Unknown')}
*Destination IP:* {alert_data.get('destination_ip', 'Unknown')}
*Protocol:* {alert_data.get('protocol', 'Unknown')}
*Confidence:* {alert_data.get('confidence', 0)*100:.1f}%
*Time:* {alert_data.get('timestamp', datetime.now())}
        """

        url = f"https://api.telegram.org/bot{self.config['telegram']['bot_token']}/sendMessage"
        payload = {
            'chat_id': self.config['telegram']['chat_id'],
            'text': message,
            'parse_mode': 'Markdown'
        }

        response = requests.post(url, json=payload, timeout=5)

        if response.status_code == 200:
            logger.info("Telegram alert sent")

    def get_notification_stats(self) -> Dict:
        """Get notification statistics"""
        total = len(self.notification_log)
        if total == 0:
            return {'total_notifications': 0}

        from collections import Counter

        priorities = Counter(log['priority'] for log in self.notification_log)
        channels = Counter(
            channel
            for log in self.notification_log
            for channel in log['channels']
        )

        return {
            'total_notifications': total,
            'by_priority': dict(priorities),
            'by_channel': dict(channels),
            'most_recent': self.notification_log[-1] if self.notification_log else None
        }


# Example usage
if __name__ == '__main__':
    print("Alert Notifier Test")
    print("=" * 70)

    # Initialize notifier (console and file only for testing)
    notifier = AlertNotifier()

    # Test alert
    alert = {
        'timestamp': datetime.now(),
        'severity': 'HIGH',
        'attack_type': 'Port Scan Attack',
        'source_ip': '192.168.1.100',
        'destination_ip': '10.0.0.50',
        'protocol': 'TCP',
        'detection_method': 'Signature + ML',
        'confidence': 0.95,
        'description': 'Multiple port access attempts detected from single source'
    }

    # Send alert
    print("\n[*] Sending test alert...\n")
    notifier.send_alert(alert, priority='HIGH')

    # Get stats
    print("\n" + "=" * 70)
    print("NOTIFICATION STATISTICS")
    print("=" * 70)
    stats = notifier.get_notification_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n✅ Notifier test completed")
