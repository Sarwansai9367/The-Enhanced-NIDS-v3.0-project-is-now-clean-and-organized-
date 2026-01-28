# Enhanced NIDS - Technical Documentation

## 📐 System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENHANCED NIDS ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│ Network Traffic  │ ◄── Raw packets (TCP/UDP/ICMP)
└────────┬─────────┘
         │
         ▼
┌─────────────────────────┐
│  Packet Capture Module  │ ◄── Captures and stores packet metadata
└────────┬────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Feature Extraction Module│ ◄── Converts raw data to numerical features
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│   Preprocessing Module   │ ◄── Cleans, normalizes, encodes data
└────────┬─────────────────┘
         │
         ├─────────────────────┬──────────────────────┐
         ▼                     ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   Signature      │  │   ML Anomaly     │  │   Hybrid         │
│   Detection      │  │   Detection      │  │   Decision       │
│   Engine         │  │   Engine         │  │   Engine         │
└──────────────────┘  └──────────────────┘  └────────┬─────────┘
         │                     │                      │
         └─────────────────────┴──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Alert & Response     │ ◄── Generates security alerts
                    │ Module               │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Performance          │ ◄── Metrics and reporting
                    │ Evaluation Module    │
                    └──────────────────────┘
```

## 🔬 Module Specifications

### 1. PacketCaptureModule

**Purpose**: Simulate network packet capture functionality

**Key Methods**:
- `capture_packet(packet_data: Dict) -> Dict`
  - Captures a single network packet
  - Adds timestamp
  - Stores packet metadata
  
- `simulate_traffic(num_packets: int) -> List[Dict]`
  - Generates synthetic network traffic
  - Simulates TCP/UDP/ICMP protocols
  - Random IP addresses and ports

**Data Structure**:
```python
packet = {
    'timestamp': datetime.now(),
    'src_ip': '192.168.1.100',
    'dst_ip': '10.0.0.50',
    'protocol': 'TCP',
    'packet_size': 1024,
    'src_port': 49152,
    'dst_port': 80,
    'flags': 'SYN'
}
```

---

### 2. FeatureExtractionModule

**Purpose**: Extract numerical features from raw packet data

**Key Methods**:
- `extract_features(packet: Dict) -> Dict`
  - Converts protocol to numerical type (TCP=0, UDP=1, ICMP=2)
  - Extracts port numbers
  - Parses flags (SYN, ACK)
  - Returns feature vector

**Feature Vector**:
```python
features = {
    'protocol_type': 0,      # TCP
    'packet_size': 1024,
    'src_port': 49152,
    'dst_port': 80,
    'flag_syn': 1,           # SYN flag present
    'flag_ack': 0            # ACK flag absent
}
```

---

### 3. PreprocessingModule

**Purpose**: Clean and normalize data for ML model

**Key Methods**:
- `remove_duplicates(df: pd.DataFrame) -> pd.DataFrame`
  - Removes duplicate entries
  - Reports removal count
  
- `handle_missing_values(df: pd.DataFrame) -> pd.DataFrame`
  - Fills numeric columns with median
  - Fills categorical columns with mode
  
- `encode_categorical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame`
  - Label encoding for categorical features
  - Maintains encoder state for consistency
  
- `normalize_features(X_train, X_test) -> Tuple`
  - StandardScaler normalization
  - Fits on training data
  - Transforms both train and test sets

**Normalization Formula**:
```
X_scaled = (X - μ) / σ
where μ = mean, σ = standard deviation
```

---

### 4. SignatureDetectionEngine

**Purpose**: Detect known attack patterns using signature matching

**Attack Signatures**:
```python
signatures = {
    'port_scan': {
        'pattern': 'multiple_ports',
        'threshold': 10
    },
    'ddos': {
        'pattern': 'high_packet_rate',
        'threshold': 100
    },
    'syn_flood': {
        'pattern': 'syn_packets',
        'threshold': 50
    },
    'known_malware_port': {
        'ports': [4444, 5555, 6666, 31337]
    }
}
```

**Detection Logic**:
1. Check destination port against malware ports
2. Detect SYN flood indicators (SYN=1, ACK=0, small packet size)
3. Identify suspicious port access (privileged ports from high ephemeral ports)

**Return**: `Tuple[bool, str]` - (is_attack, attack_type)

---

### 5. MLAnomalyDetectionEngine

**Purpose**: Machine learning-based anomaly detection

**Supported Models**:
1. **Random Forest** (Default)
   - n_estimators: 100
   - Ensemble of decision trees
   - Robust to overfitting
   
2. **Decision Tree**
   - Single tree classifier
   - Fast and interpretable
   
3. **Support Vector Machine (SVM)**
   - RBF kernel
   - Effective in high dimensions

**Key Methods**:
- `train(X_train, y_train)`
  - Fits the selected ML model
  - Reports training time
  - Sets is_trained flag
  
- `predict(X) -> np.ndarray`
  - Predicts 0 (Normal) or 1 (Intrusion)
  - Requires trained model
  
- `predict_single(features) -> Tuple[int, str]`
  - Single instance prediction
  - Returns prediction and label

**Training Process**:
```
1. Initialize model (RF/DT/SVM)
2. Fit on normalized training data
3. Evaluate on test set
4. Calculate performance metrics
```

---

### 6. HybridDetectionEngine

**Purpose**: Combine signature-based and ML-based detection

**Detection Algorithm**:
```python
FUNCTION detect(packet_features, ml_features):
    result = {
        'is_intrusion': False,
        'detection_method': [],
        'attack_type': 'Normal',
        'confidence': 0.0
    }
    
    # Step 1: Signature Detection
    IF signature_engine.detect(packet_features):
        result['is_intrusion'] = True
        result['detection_method'].append('Signature')
        result['attack_type'] = detected_type
        result['confidence'] = 0.95
    
    # Step 2: ML Detection
    IF ml_engine.predict(ml_features) == 1:
        result['is_intrusion'] = True
        result['detection_method'].append('ML')
        IF result['attack_type'] == 'Normal':
            result['attack_type'] = 'Anomaly Detected'
        result['confidence'] = max(confidence, 0.85)
    
    RETURN result
```

**Decision Logic**: OR-based (either engine flags = intrusion)

**Confidence Levels**:
- Signature detection: 95%
- ML detection: 85%
- Both methods: 95%

---

### 7. AlertResponseModule

**Purpose**: Generate and manage security alerts

**Alert Structure**:
```python
alert = {
    'alert_id': 1,
    'timestamp': datetime.now(),
    'severity': 'HIGH',  # HIGH/MEDIUM/LOW
    'attack_type': 'SYN Flood',
    'source_ip': '192.168.1.100',
    'destination_ip': '10.0.0.50',
    'protocol': 'TCP',
    'detection_method': 'Signature',
    'confidence': 0.95
}
```

**Severity Calculation**:
- **HIGH**: DDoS, SYN Flood, Known Malware
- **MEDIUM**: Port Scan, Suspicious Port
- **LOW**: Other anomalies

**Key Methods**:
- `generate_alert(detection_result, packet_info)`
  - Creates alert if intrusion detected
  - Calculates severity
  - Displays formatted alert
  - Stores in alert list
  
- `get_alert_summary() -> Dict`
  - Total alerts
  - Count by severity
  - Attack type distribution

---

### 8. PerformanceEvaluationModule

**Purpose**: Evaluate system performance and generate reports

**Performance Metrics**:

1. **Accuracy**
   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **Precision**
   ```
   Precision = TP / (TP + FP)
   ```

3. **Recall (Sensitivity)**
   ```
   Recall = TP / (TP + FN)
   ```

4. **F1 Score**
   ```
   F1 = 2 × (Precision × Recall) / (Precision + Recall)
   ```

5. **False Positive Rate**
   ```
   FPR = FP / (FP + TN)
   ```

**Confusion Matrix**:
```
                Predicted
                Normal  Intrusion
Actual Normal     TN       FP
       Intrusion  FN       TP
```

**Report Sections**:
- Detection metrics
- Confusion matrix
- Classification report
- Alert statistics
- Attack distribution

---

## 🧮 Algorithms & Formulas

### Hybrid Detection Decision Tree

```
START
  │
  ├─► Signature Check
  │   ├─ Match? ──► INTRUSION (Confidence: 95%)
  │   └─ No Match ──┐
  │                  │
  └─► ML Prediction  │
      ├─ Anomaly? ──► INTRUSION (Confidence: 85%)
      └─ Normal ────► NORMAL (Confidence: 99%)
```

### Feature Normalization

For each feature column:
```
1. Calculate mean (μ) and std (σ) from training data
2. For each value x:
   x_normalized = (x - μ) / σ
3. Apply same transformation to test data
```

### Random Forest Prediction

```
1. Create 100 decision trees
2. Each tree trained on bootstrap sample
3. For prediction:
   - Each tree votes (0 or 1)
   - Majority vote determines final class
```

---

## 📊 Data Flow

### Training Phase

```
Dataset (CSV/Synthetic)
    ↓
Load & Prepare
    ↓
Split (70% train, 30% test)
    ↓
Preprocess (normalize, encode)
    ↓
Train ML Model (Random Forest)
    ↓
Evaluate on Test Set
    ↓
Calculate Metrics
```

### Detection Phase

```
Network Packet
    ↓
Capture Metadata
    ↓
Extract Features
    ↓
Preprocess Features
    ↓
Parallel Detection
    ├─► Signature Engine
    └─► ML Engine
    ↓
Hybrid Decision (OR)
    ↓
Generate Alert (if intrusion)
    ↓
Log & Report
```

---

## 🔧 Configuration Parameters

### System Configuration

```python
# ML Model Selection
model_type = 'random_forest'  # Options: 'random_forest', 'decision_tree', 'svm'

# Dataset Configuration
n_samples = 1000              # Synthetic dataset size
test_size = 0.3               # Test set ratio (30%)
random_state = 42             # Reproducibility seed

# Random Forest Parameters
n_estimators = 100            # Number of trees
n_jobs = -1                   # Use all CPU cores

# Traffic Simulation
num_packets = 20              # Packets to process
packet_delay = 0.1            # Seconds between packets

# Detection Thresholds
signature_confidence = 0.95   # Signature detection confidence
ml_confidence = 0.85          # ML detection confidence
```

---

## 📈 Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Packet Capture | O(1) | Single packet |
| Feature Extraction | O(1) | Fixed features |
| Signature Detection | O(k) | k = number of signatures |
| ML Prediction (RF) | O(n×d) | n=trees, d=depth |
| Alert Generation | O(1) | Single alert |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Captured Packets | O(p) | p = packet count |
| ML Model (RF) | O(n×d×f) | n=trees, d=depth, f=features |
| Alert Storage | O(a) | a = alert count |
| Feature Vectors | O(f) | f = feature dimensions |

---

## 🛠️ Extension Points

### Adding Custom Signatures

```python
def _load_signatures(self):
    signatures = {
        # ... existing signatures ...
        'custom_attack': {
            'pattern': 'your_pattern',
            'threshold': threshold_value
        }
    }
    return signatures
```

### Adding New ML Model

```python
elif self.model_type == 'neural_network':
    from sklearn.neural_network import MLPClassifier
    self.model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        random_state=42
    )
```

### Custom Alert Handlers

```python
def custom_alert_handler(self, alert):
    # Send email
    # Write to database
    # Trigger automated response
    pass
```

---

## 🔬 Testing Methodology

### Unit Testing
- Test each module independently
- Verify feature extraction accuracy
- Validate preprocessing transformations
- Check signature matching logic

### Integration Testing
- Test module interactions
- Verify data flow between components
- Check hybrid decision logic
- Validate alert generation

### Performance Testing
- Measure training time
- Test detection speed
- Monitor memory usage
- Evaluate scalability

---

## 📚 References

### Datasets
- **NSL-KDD**: https://www.unb.ca/cic/datasets/nsl.html
- **CICIDS2017**: https://www.unb.ca/cic/datasets/ids-2017.html
- **KDD Cup 1999**: http://kdd.ics.uci.edu/databases/kddcup99/

### Algorithms
- Random Forest: Breiman, L. (2001)
- SVM: Cortes & Vapnik (1995)
- StandardScaler: scikit-learn documentation

### Security Standards
- NIST Cybersecurity Framework
- ISO/IEC 27001
- MITRE ATT&CK Framework

---

**Document Version**: 1.0  
**Last Updated**: January 28, 2026  
**Compatibility**: Python 3.7+, scikit-learn 1.0+
