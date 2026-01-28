"""
Real-Time Database Logger for NIDS
Implements fast database storage for packets and alerts
"""

import sqlite3
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeLogger:
    """
    High-performance database logger for real-time NIDS
    Uses connection pooling and batch inserts for speed
    """

    def __init__(self, db_path: str = 'nids_realtime.db', batch_size: int = 100):
        """
        Initialize real-time logger

        Args:
            db_path: Path to SQLite database
            batch_size: Number of records to batch before insert
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.conn = None
        self.write_queue = queue.Queue()
        self.is_logging = False
        self.writer_thread = None

        # Packet and alert batches
        self.packet_batch = []
        self.alert_batch = []

        # Statistics
        self.packets_logged = 0
        self.alerts_logged = 0

        # Initialize database
        self._init_database()
        logger.info(f"✅ Real-time logger initialized: {db_path}")

    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Packets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS packets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                src_ip TEXT NOT NULL,
                dst_ip TEXT NOT NULL,
                protocol TEXT,
                src_port INTEGER,
                dst_port INTEGER,
                packet_size INTEGER,
                flags TEXT,
                is_intrusion BOOLEAN,
                attack_type TEXT,
                confidence REAL,
                detection_method TEXT
            )
        ''')

        # Create indexes for packets table
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON packets(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_src_ip ON packets(src_ip)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dst_ip ON packets(dst_ip)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_intrusion ON packets(is_intrusion)')

        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                severity TEXT NOT NULL,
                attack_type TEXT NOT NULL,
                src_ip TEXT,
                dst_ip TEXT,
                protocol TEXT,
                detection_method TEXT,
                confidence REAL,
                description TEXT,
                mitigated BOOLEAN DEFAULT 0
            )
        ''')

        # Create indexes for alerts table
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_timestamp ON alerts(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_severity ON alerts(severity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_attack_type ON alerts(attack_type)')

        # Flow statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_key TEXT NOT NULL UNIQUE,
                src_ip TEXT,
                dst_ip TEXT,
                dst_port INTEGER,
                start_time TEXT,
                end_time TEXT,
                packet_count INTEGER DEFAULT 0,
                byte_count INTEGER DEFAULT 0,
                syn_count INTEGER DEFAULT 0,
                ack_count INTEGER DEFAULT 0,
                is_suspicious BOOLEAN DEFAULT 0
            )
        ''')

        # Create index for flows table
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_flow_key ON flows(flow_key)')

        # System statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                packets_processed INTEGER,
                intrusions_detected INTEGER,
                alerts_generated INTEGER,
                cpu_usage REAL,
                memory_usage REAL,
                detection_rate REAL
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Database tables initialized")

    def start_logging(self):
        """Start background logging thread"""
        if self.is_logging:
            logger.warning("Logging already started")
            return

        self.is_logging = True
        self.writer_thread = threading.Thread(target=self._logging_worker, daemon=True)
        self.writer_thread.start()
        logger.info("Background logging started")

    def stop_logging(self):
        """Stop logging and flush remaining data"""
        if not self.is_logging:
            return

        logger.info("Stopping logger...")
        self.is_logging = False

        # Flush remaining batches
        self._flush_batches()

        if self.writer_thread:
            self.writer_thread.join(timeout=5)

        logger.info(f"Logger stopped. Packets: {self.packets_logged}, Alerts: {self.alerts_logged}")

    def _logging_worker(self):
        """Background worker that processes write queue"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

        try:
            while self.is_logging or not self.write_queue.empty():
                try:
                    item = self.write_queue.get(timeout=1)

                    if item['type'] == 'packet':
                        self.packet_batch.append(item['data'])
                        if len(self.packet_batch) >= self.batch_size:
                            self._flush_packet_batch()

                    elif item['type'] == 'alert':
                        self.alert_batch.append(item['data'])
                        if len(self.alert_batch) >= self.batch_size // 2:  # Flush alerts faster
                            self._flush_alert_batch()

                    elif item['type'] == 'flow':
                        self._update_flow(item['data'])

                    self.write_queue.task_done()

                except queue.Empty:
                    # Flush even if batch not full
                    self._flush_batches()
                    continue

        finally:
            self._flush_batches()
            if self.conn:
                self.conn.close()

    def _flush_batches(self):
        """Flush all pending batches"""
        self._flush_packet_batch()
        self._flush_alert_batch()

    def _flush_packet_batch(self):
        """Insert packet batch into database"""
        if not self.packet_batch or not self.conn:
            return

        try:
            cursor = self.conn.cursor()
            cursor.executemany('''
                INSERT INTO packets 
                (timestamp, src_ip, dst_ip, protocol, src_port, dst_port, 
                 packet_size, flags, is_intrusion, attack_type, confidence, detection_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', self.packet_batch)

            self.conn.commit()
            self.packets_logged += len(self.packet_batch)
            self.packet_batch.clear()

        except Exception as e:
            logger.error(f"Error flushing packet batch: {e}")

    def _flush_alert_batch(self):
        """Insert alert batch into database"""
        if not self.alert_batch or not self.conn:
            return

        try:
            cursor = self.conn.cursor()
            cursor.executemany('''
                INSERT INTO alerts 
                (timestamp, severity, attack_type, src_ip, dst_ip, protocol, 
                 detection_method, confidence, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', self.alert_batch)

            self.conn.commit()
            self.alerts_logged += len(self.alert_batch)
            self.alert_batch.clear()

        except Exception as e:
            logger.error(f"Error flushing alert batch: {e}")

    def _update_flow(self, flow_data: tuple):
        """Update or insert flow statistics"""
        if not self.conn:
            return

        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO flows 
                (flow_key, src_ip, dst_ip, dst_port, start_time, end_time, 
                 packet_count, byte_count, syn_count, ack_count, is_suspicious)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(flow_key) DO UPDATE SET
                    end_time = excluded.end_time,
                    packet_count = excluded.packet_count,
                    byte_count = excluded.byte_count,
                    syn_count = excluded.syn_count,
                    ack_count = excluded.ack_count,
                    is_suspicious = excluded.is_suspicious
            ''', flow_data)
            self.conn.commit()

        except Exception as e:
            logger.error(f"Error updating flow: {e}")

    def log_packet(self, packet_data: Dict, detection_result: Optional[Dict] = None):
        """
        Log packet to database (async)

        Args:
            packet_data: Packet information dictionary
            detection_result: Detection result from hybrid engine
        """
        record = (
            packet_data.get('timestamp', datetime.now()).isoformat(),
            packet_data.get('src_ip', 'unknown'),
            packet_data.get('dst_ip', 'unknown'),
            packet_data.get('protocol', 'UNKNOWN'),
            packet_data.get('src_port', 0),
            packet_data.get('dst_port', 0),
            packet_data.get('packet_size', 0),
            packet_data.get('flags', ''),
            detection_result.get('is_intrusion', False) if detection_result else False,
            detection_result.get('attack_type', 'Normal') if detection_result else 'Normal',
            detection_result.get('confidence', 0.0) if detection_result else 0.0,
            ', '.join(detection_result.get('detection_method', [])) if detection_result else ''
        )

        self.write_queue.put({'type': 'packet', 'data': record})

    def log_alert(self, alert_data: Dict):
        """
        Log alert to database (async)

        Args:
            alert_data: Alert information dictionary
        """
        record = (
            alert_data.get('timestamp', datetime.now()).isoformat(),
            alert_data.get('severity', 'LOW'),
            alert_data.get('attack_type', 'Unknown'),
            alert_data.get('source_ip', 'unknown'),
            alert_data.get('destination_ip', 'unknown'),
            alert_data.get('protocol', 'UNKNOWN'),
            alert_data.get('detection_method', ''),
            alert_data.get('confidence', 0.0),
            alert_data.get('description', '')
        )

        self.write_queue.put({'type': 'alert', 'data': record})

    def log_flow(self, flow_key: str, flow_stats: Dict):
        """
        Log flow statistics

        Args:
            flow_key: Unique flow identifier
            flow_stats: Flow statistics dictionary
        """
        record = (
            flow_key,
            flow_stats.get('src_ip', 'unknown'),
            flow_stats.get('dst_ip', 'unknown'),
            flow_stats.get('dst_port', 0),
            flow_stats.get('start_time', datetime.now()).isoformat(),
            flow_stats.get('end_time', datetime.now()).isoformat(),
            flow_stats.get('packet_count', 0),
            flow_stats.get('byte_count', 0),
            flow_stats.get('syn_count', 0),
            flow_stats.get('ack_count', 0),
            flow_stats.get('is_suspicious', False)
        )

        self.write_queue.put({'type': 'flow', 'data': record})

    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get most recent alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT timestamp, severity, attack_type, src_ip, dst_ip, 
                   protocol, detection_method, confidence
            FROM alerts
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        columns = ['timestamp', 'severity', 'attack_type', 'src_ip', 'dst_ip',
                  'protocol', 'detection_method', 'confidence']

        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()

        return results

    def get_statistics(self, time_window: str = '1 hour') -> Dict:
        """
        Get system statistics for time window

        Args:
            time_window: Time window (e.g., '1 hour', '24 hours', '7 days')
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Parse time window
        from datetime import timedelta
        now = datetime.now()
        if 'hour' in time_window:
            hours = int(time_window.split()[0])
            cutoff = now - timedelta(hours=hours)
        elif 'day' in time_window:
            days = int(time_window.split()[0])
            cutoff = now - timedelta(days=days)
        else:
            cutoff = now - timedelta(hours=1)

        cutoff_str = cutoff.isoformat()

        # Get packet statistics
        cursor.execute('''
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN is_intrusion = 1 THEN 1 ELSE 0 END) as intrusions
            FROM packets
            WHERE timestamp >= ?
        ''', (cutoff_str,))

        packet_stats = cursor.fetchone()

        # Get alert statistics
        cursor.execute('''
            SELECT severity, COUNT(*) as count
            FROM alerts
            WHERE timestamp >= ?
            GROUP BY severity
        ''', (cutoff_str,))

        alert_stats = dict(cursor.fetchall())

        # Get top attack types
        cursor.execute('''
            SELECT attack_type, COUNT(*) as count
            FROM alerts
            WHERE timestamp >= ?
            GROUP BY attack_type
            ORDER BY count DESC
            LIMIT 5
        ''', (cutoff_str,))

        top_attacks = dict(cursor.fetchall())

        conn.close()

        return {
            'time_window': time_window,
            'total_packets': packet_stats[0] if packet_stats else 0,
            'intrusions_detected': packet_stats[1] if packet_stats else 0,
            'detection_rate': (packet_stats[1] / packet_stats[0] * 100) if packet_stats and packet_stats[0] > 0 else 0,
            'alert_severity': alert_stats,
            'top_attacks': top_attacks,
            'packets_logged': self.packets_logged,
            'alerts_logged': self.alerts_logged
        }

    def export_to_csv(self, output_file: str, table: str = 'alerts', limit: int = 1000):
        """
        Export data to CSV file

        Args:
            output_file: Output CSV file path
            table: Table name to export
            limit: Maximum number of records
        """
        import csv

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f'SELECT * FROM {table} LIMIT ?', (limit,))

        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(column_names)
            writer.writerows(rows)

        conn.close()
        logger.info(f"Exported {len(rows)} records to {output_file}")


# Example usage
if __name__ == '__main__':
    print("Real-Time Logger Test")
    print("=" * 70)

    # Initialize logger
    logger_obj = RealTimeLogger(db_path='test_nids.db', batch_size=10)
    logger_obj.start_logging()

    # Simulate logging packets
    print("\n[*] Logging 50 test packets...")
    for i in range(50):
        packet = {
            'timestamp': datetime.now(),
            'src_ip': f'192.168.1.{i % 255}',
            'dst_ip': f'10.0.0.{(i * 2) % 255}',
            'protocol': 'TCP',
            'src_port': 1024 + i,
            'dst_port': 80,
            'packet_size': 1000 + i * 10,
            'flags': 'SYN' if i % 2 == 0 else 'ACK'
        }

        detection = {
            'is_intrusion': i % 5 == 0,
            'attack_type': 'Port Scan' if i % 5 == 0 else 'Normal',
            'confidence': 0.95 if i % 5 == 0 else 0.99,
            'detection_method': ['Signature'] if i % 5 == 0 else []
        }

        logger_obj.log_packet(packet, detection)

        if i % 5 == 0:
            alert = {
                'timestamp': datetime.now(),
                'severity': 'HIGH',
                'attack_type': 'Port Scan',
                'source_ip': packet['src_ip'],
                'destination_ip': packet['dst_ip'],
                'protocol': 'TCP',
                'detection_method': 'Signature',
                'confidence': 0.95,
                'description': 'Suspicious port scanning activity detected'
            }
            logger_obj.log_alert(alert)

    # Wait for batches to flush
    import time
    time.sleep(2)

    # Get statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    stats = logger_obj.get_statistics('1 hour')

    for key, value in stats.items():
        print(f"{key}: {value}")

    # Get recent alerts
    print("\n" + "=" * 70)
    print("RECENT ALERTS")
    print("=" * 70)
    alerts = logger_obj.get_recent_alerts(5)

    for alert in alerts:
        print(f"\n🚨 {alert['severity']} - {alert['attack_type']}")
        print(f"   Source: {alert['src_ip']}")
        print(f"   Time: {alert['timestamp']}")

    # Stop logger
    logger_obj.stop_logging()

    print("\n✅ Logger test completed")
