"""
Real-Time Web Dashboard for NIDS
Flask-based web interface with live updates using WebSockets
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List
import json
import os

# Import database logger for stats
from realtime_logger import RealTimeLogger

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nids-secret-key-change-in-production'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


class DashboardController:
    """Controller for managing dashboard data and updates"""

    def __init__(self, db_path: str = 'nids_realtime.db'):
        """
        Initialize dashboard controller

        Args:
            db_path: Path to NIDS database
        """
        self.db_logger = RealTimeLogger(db_path=db_path)
        self.is_running = False
        self.update_thread = None

        # Real-time statistics
        self.current_stats = {
            'total_packets': 0,
            'intrusions': 0,
            'alerts': 0,
            'detection_rate': 0.0,
            'packets_per_second': 0.0,
            'recent_alerts': [],
            'attack_distribution': {},
            'protocol_distribution': {},
            'severity_counts': {}
        }

    def start(self):
        """Start dashboard update thread"""
        if self.is_running:
            return

        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_worker, daemon=True)
        self.update_thread.start()
        print("[+] Dashboard controller started")

    def stop(self):
        """Stop dashboard updates"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=2)

    def _update_worker(self):
        """Worker thread that periodically updates statistics"""
        while self.is_running:
            try:
                # Get updated statistics from database
                stats = self.db_logger.get_statistics('1 hour')

                # Get recent alerts
                recent_alerts = self.db_logger.get_recent_alerts(limit=10)

                # Update current stats
                self.current_stats.update({
                    'total_packets': stats.get('total_packets', 0),
                    'intrusions': stats.get('intrusions_detected', 0),
                    'alerts': stats.get('alerts_logged', 0),
                    'detection_rate': stats.get('detection_rate', 0.0),
                    'recent_alerts': recent_alerts,
                    'attack_distribution': stats.get('top_attacks', {}),
                    'severity_counts': stats.get('alert_severity', {})
                })

                # Emit update to connected clients
                socketio.emit('stats_update', self.current_stats, namespace='/dashboard')

                time.sleep(2)  # Update every 2 seconds

            except Exception as e:
                print(f"[!] Dashboard update error: {e}")
                time.sleep(5)

    def get_current_stats(self) -> Dict:
        """Get current statistics"""
        return self.current_stats

    def get_historical_data(self, hours: int = 24) -> Dict:
        """Get historical data for charts"""
        import sqlite3
        from datetime import datetime, timedelta

        try:
            conn = sqlite3.connect('nids_logs.db')
            cursor = conn.cursor()

            # Calculate cutoff time
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

            # Get historical statistics
            cursor.execute('''
                SELECT timestamp, packets_processed, intrusions_detected
                FROM system_stats
                WHERE timestamp > ?
                ORDER BY timestamp ASC
            ''', (cutoff_time,))

            rows = cursor.fetchall()
            conn.close()

            timestamps = []
            packet_counts = []
            intrusion_counts = []

            for row in rows:
                timestamps.append(row[0])
                packet_counts.append(row[1] if row[1] else 0)
                intrusion_counts.append(row[2] if row[2] else 0)

            return {
                'timestamps': timestamps,
                'packet_counts': packet_counts,
                'intrusion_counts': intrusion_counts
            }
        except Exception as e:
            print(f"[!] Error retrieving historical data: {e}")
            return {
                'timestamps': [],
                'packet_counts': [],
                'intrusion_counts': []
            }


# Global dashboard controller
dashboard = DashboardController()


# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/stats')
def get_stats():
    """API endpoint for current statistics"""
    return jsonify(dashboard.get_current_stats())


@app.route('/api/alerts')
def get_alerts():
    """API endpoint for recent alerts"""
    limit = request.args.get('limit', 20, type=int)
    alerts = dashboard.db_logger.get_recent_alerts(limit=limit)
    return jsonify(alerts)


@app.route('/api/history')
def get_history():
    """API endpoint for historical data"""
    hours = request.args.get('hours', 24, type=int)
    data = dashboard.get_historical_data(hours=hours)
    return jsonify(data)


# WebSocket events
@socketio.on('connect', namespace='/dashboard')
def handle_connect():
    """Handle client connection"""
    print('[*] Client connected to dashboard')
    emit('connection_response', {'status': 'connected'})
    emit('stats_update', dashboard.get_current_stats())


@socketio.on('disconnect', namespace='/dashboard')
def handle_disconnect():
    """Handle client disconnection"""
    print('[*] Client disconnected from dashboard')


@socketio.on('request_update', namespace='/dashboard')
def handle_update_request():
    """Handle manual update request"""
    emit('stats_update', dashboard.get_current_stats())


def create_dashboard_html():
    """Create HTML template for dashboard"""
    template_dir = 'templates'
    os.makedirs(template_dir, exist_ok=True)

    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NIDS Real-Time Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .header h1 {
            color: #667eea;
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .status {
            display: inline-block;
            padding: 5px 15px;
            background: #10b981;
            color: white;
            border-radius: 20px;
            font-size: 0.9em;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
        }
        
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-change {
            font-size: 0.85em;
            color: #10b981;
            margin-top: 5px;
        }
        
        .alerts-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .alerts-section h2 {
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .alert-item {
            padding: 15px;
            border-left: 4px solid #667eea;
            background: #f7fafc;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        
        .alert-item.high {
            border-left-color: #ef4444;
        }
        
        .alert-item.medium {
            border-left-color: #f59e0b;
        }
        
        .alert-item.low {
            border-left-color: #10b981;
        }
        
        .alert-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        
        .alert-type {
            font-weight: bold;
            color: #333;
        }
        
        .alert-severity {
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            color: white;
        }
        
        .alert-severity.HIGH {
            background: #ef4444;
        }
        
        .alert-severity.MEDIUM {
            background: #f59e0b;
        }
        
        .alert-severity.LOW {
            background: #10b981;
        }
        
        .alert-details {
            color: #666;
            font-size: 0.9em;
        }
        
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .footer {
            text-align: center;
            color: white;
            margin-top: 20px;
            font-size: 0.9em;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: white;
            font-size: 1.2em;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Network Intrusion Detection System</h1>
            <span class="status" id="status">● ACTIVE</span>
            <span style="float: right; color: #666;" id="lastUpdate">Last updated: Never</span>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Packets</div>
                <div class="stat-value" id="totalPackets">0</div>
                <div class="stat-change">Real-time monitoring</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">Intrusions Detected</div>
                <div class="stat-value" style="color: #ef4444;" id="intrusions">0</div>
                <div class="stat-change" id="detectionRate">0% detection rate</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">Alerts Generated</div>
                <div class="stat-value" style="color: #f59e0b;" id="alerts">0</div>
                <div class="stat-change">Critical monitoring</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">Packets/Second</div>
                <div class="stat-value" style="color: #10b981;" id="packetsPerSec">0</div>
                <div class="stat-change">Processing rate</div>
            </div>
        </div>
        
        <div class="alerts-section">
            <h2>🚨 Recent Security Alerts</h2>
            <div id="alertsList">
                <div class="loading pulse">Waiting for alerts...</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2 style="color: #667eea; margin-bottom: 15px;">Attack Type Distribution</h2>
            <canvas id="attackChart"></canvas>
        </div>
        
        <div class="footer">
            Enhanced Network Intrusion Detection System | Real-Time Monitoring Dashboard
        </div>
    </div>
    
    <script>
        // Initialize Socket.IO connection
        const socket = io('/dashboard');
        
        let attackChart = null;
        
        // Connection events
        socket.on('connect', () => {
            console.log('Connected to NIDS dashboard');
            document.getElementById('status').textContent = '● ACTIVE';
            document.getElementById('status').style.background = '#10b981';
        });
        
        socket.on('disconnect', () => {
            console.log('Disconnected from dashboard');
            document.getElementById('status').textContent = '● DISCONNECTED';
            document.getElementById('status').style.background = '#ef4444';
        });
        
        // Statistics update
        socket.on('stats_update', (data) => {
            updateStats(data);
        });
        
        function updateStats(stats) {
            // Update stat cards
            document.getElementById('totalPackets').textContent = 
                stats.total_packets.toLocaleString();
            document.getElementById('intrusions').textContent = 
                stats.intrusions.toLocaleString();
            document.getElementById('alerts').textContent = 
                stats.alerts.toLocaleString();
            document.getElementById('packetsPerSec').textContent = 
                stats.packets_per_second.toFixed(2);
            document.getElementById('detectionRate').textContent = 
                stats.detection_rate.toFixed(1) + '% detection rate';
            
            // Update last update time
            document.getElementById('lastUpdate').textContent = 
                'Last updated: ' + new Date().toLocaleTimeString();
            
            // Update alerts list
            updateAlertsList(stats.recent_alerts);
            
            // Update chart
            updateAttackChart(stats.attack_distribution);
        }
        
        function updateAlertsList(alerts) {
            const alertsList = document.getElementById('alertsList');
            
            if (!alerts || alerts.length === 0) {
                alertsList.innerHTML = '<div class="loading pulse">No alerts detected</div>';
                return;
            }
            
            alertsList.innerHTML = '';
            
            alerts.forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert-item ' + alert.severity.toLowerCase();
                
                alertDiv.innerHTML = `
                    <div class="alert-header">
                        <span class="alert-type">🔴 ${alert.attack_type}</span>
                        <span class="alert-severity ${alert.severity}">${alert.severity}</span>
                    </div>
                    <div class="alert-details">
                        <strong>Source:</strong> ${alert.src_ip} → 
                        <strong>Dest:</strong> ${alert.dst_ip} | 
                        <strong>Protocol:</strong> ${alert.protocol} | 
                        <strong>Time:</strong> ${new Date(alert.timestamp).toLocaleString()}
                    </div>
                `;
                
                alertsList.appendChild(alertDiv);
            });
        }
        
        function updateAttackChart(attackDist) {
            const ctx = document.getElementById('attackChart').getContext('2d');
            
            if (!attackDist || Object.keys(attackDist).length === 0) {
                return;
            }
            
            const labels = Object.keys(attackDist);
            const data = Object.values(attackDist);
            
            if (attackChart) {
                attackChart.destroy();
            }
            
            attackChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Number of Attacks',
                        data: data,
                        backgroundColor: [
                            '#ef4444',
                            '#f59e0b',
                            '#10b981',
                            '#3b82f6',
                            '#8b5cf6',
                            '#ec4899'
                        ],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                stepSize: 1
                            }
                        }
                    }
                }
            });
        }
        
        // Request initial update
        socket.emit('request_update');
        
        // Periodic manual refresh (backup)
        setInterval(() => {
            socket.emit('request_update');
        }, 5000);
    </script>
</body>
</html>'''

    with open(os.path.join(template_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"[+] Dashboard HTML template created in {template_dir}/")


def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    """
    Run the dashboard web server

    Args:
        host: Host to bind to
        port: Port to listen on
        debug: Enable debug mode
    """
    # Create HTML template
    create_dashboard_html()

    # Start dashboard controller
    dashboard.start()

    print("="*70)
    print("NIDS REAL-TIME WEB DASHBOARD")
    print("="*70)
    print(f"[+] Dashboard starting...")
    print(f"[+] Access dashboard at: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
    print(f"[+] Press Ctrl+C to stop")
    print("="*70)

    try:
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n[*] Shutting down dashboard...")
        dashboard.stop()
    except Exception as e:
        print(f"[!] Dashboard error: {e}")
        dashboard.stop()


if __name__ == '__main__':
    run_dashboard(host='0.0.0.0', port=5000, debug=False)
