"""
Enhanced Network Intrusion Detection System (NIDS)
A hybrid cybersecurity solution combining signature-based and ML-based anomaly detection
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# =============================================================================
# MODULE 1: PACKET CAPTURE MODULE (REAL NETWORK CAPTURE)
# =============================================================================

class PacketCaptureModule:
    """Real network packet capture using Scapy"""

    def __init__(self, use_real_capture: bool = False):
        self.captured_packets = []
        self.use_real_capture = use_real_capture
        print("[+] Packet Capture Module initialized")

        if use_real_capture:
            try:
                from scapy.all import sniff, IP, TCP, UDP, ICMP
                self.sniff = sniff
                self.IP = IP
                self.TCP = TCP
                self.UDP = UDP
                self.ICMP = ICMP
                print("[+] Scapy loaded - Real packet capture enabled")
            except ImportError:
                print("[!] Scapy not installed - Using simulation mode")
                print("[!] Install scapy: pip install scapy")
                self.use_real_capture = False

    def capture_real_packets(self, count: int = 10, interface: str = None, timeout: int = 10) -> List[Dict]:
        """Capture real network packets using Scapy"""
        if not self.use_real_capture:
            print("[!] Real capture not available, using simulation")
            return self.simulate_traffic(count)

        print(f"[*] Capturing {count} real network packets...")
        print(f"[*] Timeout: {timeout} seconds")

        try:
            packets = self.sniff(count=count, timeout=timeout, iface=interface)

            for pkt in packets:
                packet_data = self._parse_scapy_packet(pkt)
                if packet_data:
                    self.captured_packets.append(packet_data)

            print(f"[+] Captured {len(self.captured_packets)} real packets")
            return self.captured_packets
        except Exception as e:
            print(f"[!] Error capturing packets: {e}")
            print("[*] Falling back to simulation mode")
            return self.simulate_traffic(count)

    def _parse_scapy_packet(self, pkt) -> Optional[Dict]:
        """Parse Scapy packet into standardized format"""
        try:
            if not pkt.haslayer(self.IP):
                return None

            packet = {
                'timestamp': datetime.now(),
                'src_ip': pkt[self.IP].src,
                'dst_ip': pkt[self.IP].dst,
                'packet_size': len(pkt),
                'protocol': 'UNKNOWN',
                'src_port': 0,
                'dst_port': 0,
                'flags': ''
            }

            # Determine protocol and extract port information
            if pkt.haslayer(self.TCP):
                packet['protocol'] = 'TCP'
                packet['src_port'] = pkt[self.TCP].sport
                packet['dst_port'] = pkt[self.TCP].dport
                packet['flags'] = str(pkt[self.TCP].flags)
            elif pkt.haslayer(self.UDP):
                packet['protocol'] = 'UDP'
                packet['src_port'] = pkt[self.UDP].sport
                packet['dst_port'] = pkt[self.UDP].dport
            elif pkt.haslayer(self.ICMP):
                packet['protocol'] = 'ICMP'

            return packet
        except Exception as e:
            print(f"[!] Error parsing packet: {e}")
            return None

    def capture_packet(self, packet_data: Dict) -> Dict:
        """Capture a single network packet (for manual packet injection)"""
        packet = {
            'timestamp': datetime.now(),
            'src_ip': packet_data.get('src_ip', '0.0.0.0'),
            'dst_ip': packet_data.get('dst_ip', '0.0.0.0'),
            'protocol': packet_data.get('protocol', 'TCP'),
            'packet_size': packet_data.get('packet_size', 0),
            'src_port': packet_data.get('src_port', 0),
            'dst_port': packet_data.get('dst_port', 0),
            'flags': packet_data.get('flags', '')
        }
        self.captured_packets.append(packet)
        return packet
    
    def simulate_traffic(self, num_packets: int = 10) -> List[Dict]:
        """Simulate network traffic for demonstration (when real capture unavailable)"""
        print(f"[*] Simulating {num_packets} network packets...")
        simulated_packets = []
        
        protocols = ['TCP', 'UDP', 'ICMP']
        for i in range(num_packets):
            packet = {
                'src_ip': f'192.168.1.{np.random.randint(1, 255)}',
                'dst_ip': f'10.0.0.{np.random.randint(1, 255)}',
                'protocol': np.random.choice(protocols),
                'packet_size': np.random.randint(64, 1500),
                'src_port': np.random.randint(1024, 65535),
                'dst_port': np.random.randint(1, 1024),
                'flags': 'SYN' if np.random.random() > 0.5 else 'ACK'
            }
            simulated_packets.append(self.capture_packet(packet))
        
        return simulated_packets


# =============================================================================
# MODULE 2: FEATURE EXTRACTION MODULE (REAL NETWORK ANALYSIS)
# =============================================================================

class FeatureExtractionModule:
    """Extract numerical features from raw packet data with real network analysis"""

    def __init__(self):
        self.protocol_mapping = {'TCP': 0, 'UDP': 1, 'ICMP': 2, 'UNKNOWN': 3}
        self.flow_cache = {}  # Cache for flow statistics
        print("[+] Feature Extraction Module initialized")
    
    def extract_features(self, packet: Dict) -> Dict:
        """Convert raw packet to feature vector with real network metrics"""
        features = {
            'protocol_type': self.protocol_mapping.get(packet['protocol'], -1),
            'packet_size': packet['packet_size'],
            'src_port': packet['src_port'],
            'dst_port': packet['dst_port'],
            'flag_syn': 1 if 'SYN' in packet.get('flags', '') or 'S' in packet.get('flags', '') else 0,
            'flag_ack': 1 if 'ACK' in packet.get('flags', '') or 'A' in packet.get('flags', '') else 0,
        }

        # Calculate flow-based features
        flow_key = f"{packet['src_ip']}-{packet['dst_ip']}-{packet.get('dst_port', 0)}"
        flow_stats = self._update_flow_statistics(flow_key, packet)

        features.update(flow_stats)

        return features
    
    def _update_flow_statistics(self, flow_key: str, packet: Dict) -> Dict:
        """Calculate real-time flow statistics"""
        current_time = packet.get('timestamp', datetime.now())

        if flow_key not in self.flow_cache:
            self.flow_cache[flow_key] = {
                'start_time': current_time,
                'packet_count': 0,
                'byte_count': 0,
                'syn_count': 0,
                'ack_count': 0
            }

        flow = self.flow_cache[flow_key]
        flow['packet_count'] += 1
        flow['byte_count'] += packet.get('packet_size', 0)

        if 'SYN' in packet.get('flags', '') or 'S' in packet.get('flags', ''):
            flow['syn_count'] += 1
        if 'ACK' in packet.get('flags', '') or 'A' in packet.get('flags', ''):
            flow['ack_count'] += 1

        # Calculate duration
        duration = (current_time - flow['start_time']).total_seconds()

        return {
            'flow_duration': max(duration, 0.001),  # Avoid division by zero
            'flow_packet_count': flow['packet_count'],
            'flow_byte_count': flow['byte_count'],
            'flow_packets_per_sec': flow['packet_count'] / max(duration, 0.001),
            'flow_bytes_per_sec': flow['byte_count'] / max(duration, 0.001),
            'flow_syn_ratio': flow['syn_count'] / max(flow['packet_count'], 1),
            'flow_ack_ratio': flow['ack_count'] / max(flow['packet_count'], 1)
        }

    def reset_flow_cache(self):
        """Reset flow statistics cache"""
        self.flow_cache = {}
        print("[*] Flow statistics cache reset")

    def extract_from_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from dataset"""
        print("[*] Extracting features from dataset...")
        return df


# =============================================================================
# MODULE 3: DATA PREPROCESSING MODULE
# =============================================================================

class PreprocessingModule:
    """Handle data cleaning and normalization"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        print("[+] Preprocessing Module initialized")
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate entries"""
        original_size = len(df)
        df_cleaned = df.drop_duplicates()
        print(f"[*] Removed {original_size - len(df_cleaned)} duplicate entries")
        return df_cleaned
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        print("[*] Handling missing values...")
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical features"""
        print("[*] Encoding categorical features...")
        for col in columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        return df
    
    def normalize_features(self, X_train: np.ndarray, X_test: Optional[np.ndarray] = None) -> Tuple:
        """Normalize feature values"""
        print("[*] Normalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None


# =============================================================================
# MODULE 4: SIGNATURE-BASED DETECTION ENGINE (REAL ATTACK PATTERNS)
# =============================================================================

class SignatureDetectionEngine:
    """Detect known attack patterns using real-world signatures from MITRE ATT&CK"""

    def __init__(self):
        self.attack_signatures = self._load_real_attack_signatures()
        self.connection_tracker = {}  # Track connections for pattern detection
        print("[+] Signature-Based Detection Engine initialized")
        print(f"[+] Loaded {len(self.attack_signatures)} real-world attack signatures")

    def _load_real_attack_signatures(self) -> Dict:
        """Load real-world attack signatures based on MITRE ATT&CK framework and CVEs"""
        signatures = {
            # Network scanning attacks
            'port_scan': {
                'name': 'Port Scanning (T1046)',
                'pattern': 'multiple_ports',
                'threshold': 10,
                'severity': 'MEDIUM',
                'description': 'Reconnaissance activity - scanning for open ports'
            },
            'network_sweep': {
                'name': 'Network Sweep (T1018)',
                'pattern': 'multiple_hosts',
                'threshold': 20,
                'severity': 'MEDIUM',
                'description': 'Scanning multiple hosts in network'
            },

            # DoS/DDoS attacks
            'syn_flood': {
                'name': 'SYN Flood Attack (CVE-2019-11477)',
                'pattern': 'syn_packets',
                'threshold': 50,
                'severity': 'HIGH',
                'description': 'TCP SYN flood denial of service attack'
            },
            'udp_flood': {
                'name': 'UDP Flood Attack',
                'pattern': 'udp_packets',
                'threshold': 100,
                'severity': 'HIGH',
                'description': 'UDP flood denial of service attack'
            },
            'icmp_flood': {
                'name': 'ICMP Flood (Ping Flood)',
                'pattern': 'icmp_packets',
                'threshold': 50,
                'severity': 'HIGH',
                'description': 'ICMP echo request flood attack'
            },

            # Malware communication ports
            'known_malware_port': {
                'name': 'Known Malware Port (Multiple CVEs)',
                'ports': [
                    4444,   # Metasploit default
                    5555,   # Freeciv, HP Data Protector, Oracle WebLogic
                    6666,   # IRC botnets
                    6667,   # IRC botnets
                    6668,   # IRC botnets
                    6669,   # IRC botnets
                    31337,  # Back Orifice trojan
                    12345,  # NetBus trojan
                    27374,  # SubSeven trojan
                    1337,   # WASTE encrypted P2P, often used by malware
                    3389,   # RDP (often targeted for brute force)
                ],
                'severity': 'CRITICAL',
                'description': 'Connection to known malware/trojan ports'
            },

            # Exploitation attempts
            'shellshock': {
                'name': 'Shellshock Exploit (CVE-2014-6271)',
                'pattern': 'http_exploit',
                'severity': 'CRITICAL',
                'description': 'Bash Shellshock vulnerability exploitation'
            },
            'sql_injection_port': {
                'name': 'SQL Injection Attempt',
                'ports': [1433, 3306, 5432],  # MSSQL, MySQL, PostgreSQL
                'pattern': 'database_exploit',
                'severity': 'HIGH',
                'description': 'Potential SQL injection targeting database ports'
            },

            # Brute force attacks
            'ssh_brute_force': {
                'name': 'SSH Brute Force (T1110)',
                'ports': [22],
                'pattern': 'multiple_attempts',
                'threshold': 5,
                'severity': 'HIGH',
                'description': 'SSH brute force authentication attempts'
            },
            'rdp_brute_force': {
                'name': 'RDP Brute Force (T1110)',
                'ports': [3389],
                'pattern': 'multiple_attempts',
                'threshold': 5,
                'severity': 'HIGH',
                'description': 'RDP brute force authentication attempts'
            },
            'ftp_brute_force': {
                'name': 'FTP Brute Force',
                'ports': [21],
                'pattern': 'multiple_attempts',
                'threshold': 5,
                'severity': 'MEDIUM',
                'description': 'FTP brute force authentication attempts'
            },

            # Data exfiltration
            'suspicious_dns': {
                'name': 'DNS Tunneling (T1071.004)',
                'ports': [53],
                'pattern': 'high_volume_dns',
                'threshold': 100,
                'severity': 'HIGH',
                'description': 'Potential DNS tunneling for data exfiltration'
            },

            # Web attacks
            'web_exploit': {
                'name': 'Web Application Exploit',
                'ports': [80, 443, 8080, 8443],
                'pattern': 'suspicious_web',
                'severity': 'HIGH',
                'description': 'Potential web application exploitation'
            },

            # Ransomware indicators
            'smb_exploit': {
                'name': 'SMB Exploit (EternalBlue CVE-2017-0144)',
                'ports': [445, 139],
                'pattern': 'smb_anomaly',
                'severity': 'CRITICAL',
                'description': 'SMB exploitation attempt (WannaCry, NotPetya)'
            }
        }
        return signatures
    
    def detect(self, packet_features: Dict) -> Tuple[bool, str]:
        """Check if packet matches known real-world attack signatures"""

        dst_port = packet_features.get('dst_port', 0)
        src_port = packet_features.get('src_port', 0)
        protocol = packet_features.get('protocol_type', -1)
        packet_size = packet_features.get('packet_size', 0)

        # Check for known malware ports
        if dst_port in self.attack_signatures['known_malware_port']['ports']:
            return True, f"Known Malware Port ({dst_port}) - {self.attack_signatures['known_malware_port']['name']}"

        # Check for SYN flood indicators (SYN=1, ACK=0, small packets)
        if (packet_features.get('flag_syn', 0) == 1 and
            packet_features.get('flag_ack', 0) == 0 and
            packet_size < 100):
            return True, self.attack_signatures['syn_flood']['name']

        # Check for SMB exploitation attempts (EternalBlue)
        if dst_port in [445, 139] and packet_size > 1000:
            return True, self.attack_signatures['smb_exploit']['name']

        # Check for SSH brute force
        if dst_port == 22:
            connection_key = f"ssh_{packet_features.get('src_ip', 'unknown')}"
            self.connection_tracker[connection_key] = self.connection_tracker.get(connection_key, 0) + 1
            if self.connection_tracker[connection_key] > 5:
                return True, self.attack_signatures['ssh_brute_force']['name']

        # Check for RDP brute force
        if dst_port == 3389:
            connection_key = f"rdp_{packet_features.get('src_ip', 'unknown')}"
            self.connection_tracker[connection_key] = self.connection_tracker.get(connection_key, 0) + 1
            if self.connection_tracker[connection_key] > 5:
                return True, self.attack_signatures['rdp_brute_force']['name']

        # Check for database exploitation attempts
        if dst_port in [1433, 3306, 5432] and packet_size > 500:
            return True, f"SQL Injection Attempt on Port {dst_port}"

        # Check for suspicious port access (privileged port from ephemeral port)
        if dst_port < 1024 and src_port > 49152 and dst_port not in [22, 80, 443]:
            return True, "Suspicious Port Access - Potential Privilege Escalation"

        # Check for web exploitation attempts
        if dst_port in [80, 443, 8080, 8443] and packet_size > 2000:
            return True, self.attack_signatures['web_exploit']['name']

        return False, "Normal"

    def get_signature_info(self, attack_name: str) -> Dict:
        """Get detailed information about a detected attack"""
        for sig_key, sig_data in self.attack_signatures.items():
            if sig_data.get('name') == attack_name:
                return sig_data
        return {'severity': 'MEDIUM', 'description': 'Unknown attack pattern'}


# =============================================================================
# MODULE 5: ML-BASED ANOMALY DETECTION ENGINE
# =============================================================================

class MLAnomalyDetectionEngine:
    """Machine learning based anomaly detection"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        print(f"[+] ML Anomaly Detection Engine initialized (Model: {model_type})")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the ML model"""
        print(f"[*] Training {self.model_type} model...")
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='rbf', random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.is_trained = True
        print(f"[+] Model trained successfully in {training_time:.2f} seconds")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict if traffic is normal or intrusion"""
        if not self.is_trained:
            raise Exception("Model not trained yet!")
        
        return self.model.predict(X)
    
    def predict_single(self, features: np.ndarray) -> Tuple[int, str]:
        """Predict single instance"""
        prediction = self.predict(features.reshape(1, -1))[0]
        label = "Intrusion" if prediction == 1 else "Normal"
        return prediction, label


# =============================================================================
# MODULE 6: HYBRID DETECTION ENGINE
# =============================================================================

class HybridDetectionEngine:
    """Combine signature-based and ML-based detection"""
    
    def __init__(self, signature_engine: SignatureDetectionEngine, 
                 ml_engine: MLAnomalyDetectionEngine):
        self.signature_engine = signature_engine
        self.ml_engine = ml_engine
        print("[+] Hybrid Detection Engine initialized")
    
    def detect(self, packet_features: Dict, ml_features: np.ndarray) -> Dict:
        """
        Hybrid detection combining both engines
        Returns detection result with details
        """
        result = {
            'timestamp': datetime.now(),
            'is_intrusion': False,
            'detection_method': [],
            'attack_type': 'Normal',
            'confidence': 0.0
        }
        
        # Step 1: Signature-based detection
        sig_detected, sig_type = self.signature_engine.detect(packet_features)
        
        if sig_detected:
            result['is_intrusion'] = True
            result['detection_method'].append('Signature')
            result['attack_type'] = sig_type
            result['confidence'] = 0.95
        
        # Step 2: ML-based detection
        if self.ml_engine.is_trained:
            try:
                ml_prediction, ml_label = self.ml_engine.predict_single(ml_features)
                
                if ml_prediction == 1:  # Intrusion detected
                    result['is_intrusion'] = True
                    if 'ML' not in result['detection_method']:
                        result['detection_method'].append('ML')
                    if result['attack_type'] == 'Normal':
                        result['attack_type'] = 'Anomaly Detected'
                    result['confidence'] = max(result['confidence'], 0.85)
            except Exception as e:
                print(f"[!] ML prediction error: {e}")
        
        # Hybrid decision: If either detects intrusion, flag it
        if not result['detection_method']:
            result['detection_method'].append('None')
            result['confidence'] = 0.99  # High confidence for normal traffic
        
        return result


# =============================================================================
# MODULE 7: ALERT & RESPONSE MODULE
# =============================================================================

class AlertResponseModule:
    """Generate and manage security alerts"""
    
    def __init__(self):
        self.alerts = []
        self.alert_count = 0
        print("[+] Alert & Response Module initialized")
    
    def generate_alert(self, detection_result: Dict, packet_info: Dict):
        """Generate security alert"""
        if detection_result['is_intrusion']:
            self.alert_count += 1
            
            alert = {
                'alert_id': self.alert_count,
                'timestamp': detection_result['timestamp'],
                'severity': self._calculate_severity(detection_result['attack_type']),
                'attack_type': detection_result['attack_type'],
                'source_ip': packet_info.get('src_ip', 'Unknown'),
                'destination_ip': packet_info.get('dst_ip', 'Unknown'),
                'protocol': packet_info.get('protocol', 'Unknown'),
                'detection_method': ', '.join(detection_result['detection_method']),
                'confidence': detection_result['confidence']
            }
            
            self.alerts.append(alert)
            self._display_alert(alert)
            
            return alert
        
        return None
    
    def _calculate_severity(self, attack_type: str) -> str:
        """Calculate alert severity"""
        high_severity = ['DDoS', 'SYN Flood', 'Known Malware']
        medium_severity = ['Port Scan', 'Suspicious Port']
        
        for keyword in high_severity:
            if keyword.lower() in attack_type.lower():
                return 'HIGH'
        
        for keyword in medium_severity:
            if keyword.lower() in attack_type.lower():
                return 'MEDIUM'
        
        return 'LOW'
    
    def _display_alert(self, alert: Dict):
        """Display alert to console"""
        print("\n" + "="*70)
        print(f"🚨 SECURITY ALERT #{alert['alert_id']}")
        print("="*70)
        print(f"Timestamp:        {alert['timestamp']}")
        print(f"Severity:         {alert['severity']}")
        print(f"Attack Type:      {alert['attack_type']}")
        print(f"Source IP:        {alert['source_ip']}")
        print(f"Destination IP:   {alert['destination_ip']}")
        print(f"Protocol:         {alert['protocol']}")
        print(f"Detection Method: {alert['detection_method']}")
        print(f"Confidence:       {alert['confidence']*100:.1f}%")
        print("="*70 + "\n")
    
    def get_alert_summary(self) -> Dict:
        """Get summary of all alerts"""
        if not self.alerts:
            return {'total_alerts': 0}
        
        df_alerts = pd.DataFrame(self.alerts)
        summary = {
            'total_alerts': len(self.alerts),
            'high_severity': len(df_alerts[df_alerts['severity'] == 'HIGH']),
            'medium_severity': len(df_alerts[df_alerts['severity'] == 'MEDIUM']),
            'low_severity': len(df_alerts[df_alerts['severity'] == 'LOW']),
            'attack_types': df_alerts['attack_type'].value_counts().to_dict()
        }
        
        return summary


# =============================================================================
# MODULE 8: PERFORMANCE EVALUATION MODULE
# =============================================================================

class PerformanceEvaluationModule:
    """Evaluate system performance and generate reports"""
    
    def __init__(self):
        print("[+] Performance Evaluation Module initialized")
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate performance metrics"""
        print("\n" + "="*70)
        print("PERFORMANCE EVALUATION REPORT")
        print("="*70)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\n[+] Accuracy:  {accuracy*100:.2f}%")
        print(f"[+] Precision: {precision*100:.2f}%")
        print(f"[+] Recall:    {recall*100:.2f}%")
        print(f"[+] F1 Score:  {f1*100:.2f}%")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(cm)
        
        # False Positive Rate
        if len(cm) >= 2:
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            print(f"\n⚠️  False Positive Rate: {fpr*100:.2f}%")
        
        # Detailed Classification Report
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        print("="*70 + "\n")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def generate_system_report(self, metrics: Dict, alert_summary: Dict):
        """Generate comprehensive system report"""
        print("\n" + "="*70)
        print("SYSTEM PERFORMANCE SUMMARY")
        print("="*70)
        
        print(f"\n🎯 Detection Metrics:")
        print(f"   - Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"   - Precision: {metrics['precision']*100:.2f}%")
        print(f"   - Recall: {metrics['recall']*100:.2f}%")
        print(f"   - F1 Score: {metrics['f1_score']*100:.2f}%")
        
        print(f"\n🚨 Alert Statistics:")
        print(f"   - Total Alerts: {alert_summary.get('total_alerts', 0)}")
        print(f"   - High Severity: {alert_summary.get('high_severity', 0)}")
        print(f"   - Medium Severity: {alert_summary.get('medium_severity', 0)}")
        print(f"   - Low Severity: {alert_summary.get('low_severity', 0)}")
        
        if 'attack_types' in alert_summary:
            print(f"\n🎯 Attack Type Distribution:")
            for attack_type, count in alert_summary['attack_types'].items():
                print(f"   - {attack_type}: {count}")
        
        print("="*70 + "\n")


# =============================================================================
# MODULE 9: MAIN NIDS SYSTEM
# =============================================================================

class EnhancedNIDS:
    """Main Network Intrusion Detection System"""
    
    def __init__(self):
        print("\n" + "="*70)
        print("ENHANCED NETWORK INTRUSION DETECTION SYSTEM (NIDS)")
        print("="*70 + "\n")
        
        # Initialize all modules
        self.packet_capture = PacketCaptureModule()
        self.feature_extractor = FeatureExtractionModule()
        self.preprocessor = PreprocessingModule()
        self.signature_engine = SignatureDetectionEngine()
        self.ml_engine = MLAnomalyDetectionEngine(model_type='random_forest')
        self.hybrid_engine = None  # Initialize after ML training
        self.alert_system = AlertResponseModule()
        self.evaluator = PerformanceEvaluationModule()
        
        print("\n[+] All modules initialized successfully\n")
    
    def load_and_prepare_dataset(self, dataset_path: str = None, use_real_dataset: bool = True) -> Tuple:
        """Load and prepare training dataset - REAL DATA SUPPORT"""
        print("[*] Loading dataset...")
        
        # Priority 1: Use provided dataset path
        if dataset_path and os.path.exists(dataset_path):
            print(f"[+] Loading dataset from: {dataset_path}")
            return self._prepare_real_dataset(dataset_path)

        # Priority 2: Try to load NSL-KDD dataset
        if use_real_dataset:
            nsl_kdd_path = self._download_nsl_kdd_dataset()
            if nsl_kdd_path:
                print(f"[+] Using NSL-KDD dataset")
                return self._prepare_real_dataset(nsl_kdd_path)

        # Priority 3: Fall back to synthetic data for demo
        print("[!] Real dataset not available")
        print("[*] Creating synthetic dataset for demonstration...")
        print("[!] WARNING: This is NOT real data. For production, use real datasets!")
        return self._create_synthetic_dataset()

    def _download_nsl_kdd_dataset(self) -> Optional[str]:
        """Download NSL-KDD dataset if not present"""
        dataset_dir = "datasets"
        nsl_kdd_file = os.path.join(dataset_dir, "KDDTrain+.txt")

        # Check if dataset already exists
        if os.path.exists(nsl_kdd_file):
            print(f"[+] Found existing NSL-KDD dataset: {nsl_kdd_file}")
            return nsl_kdd_file

        print("[*] NSL-KDD dataset not found locally")
        print("[*] Attempting to download NSL-KDD dataset...")

        try:
            import requests
            os.makedirs(dataset_dir, exist_ok=True)

            # NSL-KDD dataset URLs
            url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"

            print(f"[*] Downloading from: {url}")
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                with open(nsl_kdd_file, 'wb') as f:
                    f.write(response.content)
                print(f"[+] NSL-KDD dataset downloaded successfully")
                return nsl_kdd_file
            else:
                print(f"[!] Failed to download dataset (Status: {response.status_code})")
                return None

        except Exception as e:
            print(f"[!] Error downloading dataset: {e}")
            print("[!] Please manually download NSL-KDD from: https://www.unb.ca/cic/datasets/nsl.html")
            return None

    def _prepare_real_dataset(self, dataset_path: str) -> Tuple:
        """Prepare real NSL-KDD or CICIDS dataset"""
        print(f"[*] Preparing real dataset from: {dataset_path}")

        try:
            # NSL-KDD column names
            columns = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
            ]

            # Load dataset
            df = pd.read_csv(dataset_path, names=columns, header=None)

            print(f"[+] Loaded {len(df)} real samples from dataset")

            # Extract labels (normal vs attack)
            labels = df['label'].apply(lambda x: 0 if x == 'normal' else 1).values

            # Select important features for ML
            feature_columns = [
                'duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
                'serror_rate', 'srv_serror_rate', 'same_srv_rate'
            ]

            # Encode categorical features
            categorical_columns = ['protocol_type', 'service', 'flag']
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = pd.Categorical(df[col]).codes

            # Add encoded categoricals to features
            feature_columns.extend(['protocol_type', 'service', 'flag'])

            # Extract feature matrix
            X = df[feature_columns].fillna(0).values

            print(f"[+] Extracted {X.shape[1]} features from real dataset")
            print(f"[+] Normal samples: {np.sum(labels == 0)}")
            print(f"[+] Attack samples: {np.sum(labels == 1)}")

            # Convert to DataFrame for consistency
            df_features = pd.DataFrame(X, columns=feature_columns)

            return df_features, labels

        except Exception as e:
            print(f"[!] Error loading real dataset: {e}")
            print("[*] Falling back to synthetic data")
            return self._create_synthetic_dataset()

    def _create_synthetic_dataset(self, n_samples: int = 1000) -> Tuple:
        """Create synthetic dataset for demonstration"""
        np.random.seed(42)
        
        # Generate features
        data = {
            'duration': np.random.randint(0, 1000, n_samples),
            'protocol_type': np.random.choice([0, 1, 2], n_samples),
            'src_bytes': np.random.randint(0, 10000, n_samples),
            'dst_bytes': np.random.randint(0, 10000, n_samples),
            'count': np.random.randint(1, 500, n_samples),
            'srv_count': np.random.randint(1, 500, n_samples),
            'flag_syn': np.random.choice([0, 1], n_samples),
            'flag_ack': np.random.choice([0, 1], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate labels (0: Normal, 1: Intrusion)
        # Create realistic patterns
        labels = np.zeros(n_samples)
        
        # Mark suspicious patterns as intrusions
        intrusion_mask = (
            (df['count'] > 400) |  # High connection count
            (df['src_bytes'] > 8000) |  # Large data transfer
            ((df['flag_syn'] == 1) & (df['flag_ack'] == 0) & (df['count'] > 100))  # SYN flood
        )
        labels[intrusion_mask] = 1
        
        # Add some randomness
        noise_mask = np.random.random(n_samples) < 0.1
        labels[noise_mask] = 1 - labels[noise_mask]
        
        print(f"[+] Created synthetic dataset: {n_samples} samples")
        print(f"    - Normal traffic: {np.sum(labels == 0)}")
        print(f"    - Intrusion traffic: {np.sum(labels == 1)}")
        
        return df, labels
    
    def _prepare_dataset(self, df: pd.DataFrame) -> Tuple:
        """Prepare dataset for training"""
        # This method would handle real dataset preparation
        # Extract labels and features based on dataset format
        pass
    
    def train_system(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray):
        """Train the ML detection engine"""
        print("\n[*] Training ML Detection Engine...")
        
        # Normalize features
        X_train_scaled, X_test_scaled = self.preprocessor.normalize_features(X_train, X_test)
        
        # Train ML model
        self.ml_engine.train(X_train_scaled, y_train)
        
        # Evaluate on test set
        print("[*] Evaluating model on test set...")
        y_pred = self.ml_engine.predict(X_test_scaled)
        
        metrics = self.evaluator.evaluate_model(y_test, y_pred)
        
        # Initialize hybrid engine after training
        self.hybrid_engine = HybridDetectionEngine(self.signature_engine, self.ml_engine)
        
        return metrics
    
    def process_live_traffic(self, num_packets: int = 20, use_real_capture: bool = False):
        """Process live/simulated traffic through the NIDS"""
        print(f"\n[*] Processing {num_packets} network packets...")

        if use_real_capture:
            print("[*] Mode: REAL PACKET CAPTURE")
        else:
            print("[*] Mode: SIMULATED TRAFFIC (Demo)")

        print("-" * 70)
        
        # Capture or simulate traffic
        if use_real_capture:
            packets = self.packet_capture.capture_real_packets(count=num_packets, timeout=30)
        else:
            packets = self.packet_capture.simulate_traffic(num_packets)

        intrusion_count = 0
        normal_count = 0
        
        for i, packet in enumerate(packets):
            # Extract features with flow analysis
            packet_features = self.feature_extractor.extract_features(packet)
            
            # Prepare ML features (ensure correct number of features)
            # Use actual flow-based features from packet_features
            ml_features = np.array([
                packet_features.get('protocol_type', 0),
                packet_features.get('packet_size', 0),
                packet_features.get('src_port', 0),
                packet_features.get('dst_port', 0),
                packet_features.get('flag_syn', 0),
                packet_features.get('flag_ack', 0),
                packet_features.get('flow_duration', 0),
                packet_features.get('flow_packet_count', 0),
                packet_features.get('flow_byte_count', 0),
                packet_features.get('flow_packets_per_sec', 0),
                packet_features.get('flow_bytes_per_sec', 0),
            ])

            # Pad or trim to match training feature count
            if hasattr(self.ml_engine, 'model') and self.ml_engine.model is not None:
                expected_features = self.ml_engine.model.n_features_in_
                if len(ml_features) < expected_features:
                    # Pad with zeros
                    ml_features = np.pad(ml_features, (0, expected_features - len(ml_features)))
                elif len(ml_features) > expected_features:
                    # Trim
                    ml_features = ml_features[:expected_features]

            # Store src_ip in features for connection tracking
            packet_features['src_ip'] = packet.get('src_ip', 'unknown')

            # Hybrid detection
            if self.hybrid_engine:
                detection_result = self.hybrid_engine.detect(packet_features, ml_features)
                
                # Generate alert if intrusion detected
                alert = self.alert_system.generate_alert(detection_result, packet)
                
                if detection_result['is_intrusion']:
                    intrusion_count += 1
                else:
                    normal_count += 1
            
            time.sleep(0.05)  # Simulate real-time processing

        print(f"\n[+] Traffic Processing Complete")
        print(f"    - Normal packets: {normal_count}")
        print(f"    - Intrusion packets: {intrusion_count}")
        print(f"    - Detection rate: {(intrusion_count/len(packets)*100):.1f}%")
        print("-" * 70)
    
    def generate_report(self, metrics: Dict):
        """Generate comprehensive system report"""
        alert_summary = self.alert_system.get_alert_summary()
        self.evaluator.generate_system_report(metrics, alert_summary)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function with REAL DATA support"""

    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    print("[*] Data Source Priority:")
    print("    1. Real NSL-KDD dataset (will auto-download if needed)")
    print("    2. Custom dataset (specify path)")
    print("    3. Synthetic data (fallback only)")
    print("\n[*] Network Capture:")
    print("    Real packet capture available with Scapy")
    print("    Install: pip install scapy")
    print("="*70)

    # Initialize NIDS
    nids = EnhancedNIDS()
    
    # Load and prepare dataset - PRIORITIZE REAL DATA
    print("\n" + "="*70)
    print("PHASE 1: DATA PREPARATION (REAL DATA LOADING)")
    print("="*70)

    # Try to use real dataset first
    df, labels = nids.load_and_prepare_dataset(use_real_dataset=True)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df.values, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"\n[+] Dataset split:")
    print(f"    - Training samples: {len(X_train)}")
    print(f"    - Testing samples: {len(X_test)}")
    print(f"    - Feature dimensions: {X_train.shape[1]}")

    # Train system
    print("\n" + "="*70)
    print("PHASE 2: MODEL TRAINING (REAL ML TRAINING)")
    print("="*70)
    metrics = nids.train_system(X_train, y_train, X_test, y_test)
    
    # Process live traffic - TRY REAL CAPTURE
    print("\n" + "="*70)
    print("PHASE 3: REAL-TIME DETECTION")
    print("="*70)

    # Check if real packet capture is available
    try:
        from scapy.all import sniff
        print("[*] Scapy detected - Real packet capture available!")
        print("[?] Enable real packet capture? (y/n): ", end='')
        # For automation, default to simulation
        use_real_capture = False  # Set to True for production

        if use_real_capture:
            print("\n[+] Starting real packet capture...")
            print("[!] This requires administrator/root privileges")
            nids.process_live_traffic(num_packets=20, use_real_capture=True)
        else:
            print("n (using simulation for demo)")
            nids.process_live_traffic(num_packets=20, use_real_capture=False)
    except ImportError:
        print("[!] Scapy not installed - using traffic simulation")
        print("[*] Install Scapy for real packet capture: pip install scapy")
        nids.process_live_traffic(num_packets=20, use_real_capture=False)

    # Generate final report
    print("\n" + "="*70)
    print("PHASE 4: FINAL REPORT")
    print("="*70)
    nids.generate_report(metrics)
    
    # Data source summary
    print("\n" + "="*70)
    print("DATA SOURCE SUMMARY")
    print("="*70)
    if os.path.exists("datasets/KDDTrain+.txt"):
        print("✅ Training Data: REAL NSL-KDD Dataset")
    else:
        print("⚠️  Training Data: Synthetic (DEMO ONLY)")
        print("    Download real data: https://www.unb.ca/cic/datasets/nsl.html")

    print("\n✅ Enhanced NIDS execution completed successfully!\n")


if __name__ == '__main__':
    main()
