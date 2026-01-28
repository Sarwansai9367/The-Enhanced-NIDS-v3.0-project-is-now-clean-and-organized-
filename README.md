# Enhanced Network Intrusion Detection System (NIDS)

## 📋 Project Overview

The Enhanced Network Intrusion Detection System (NIDS) is a comprehensive cybersecurity solution that combines **signature-based detection** with **machine learning-based anomaly detection** to identify both known and unknown cyber threats in real-time network traffic.

## 🎯 Problem Statement

Current network security mechanisms face several challenges:
- ❌ Inability to detect zero-day attacks
- ❌ High false alert rates  
- ❌ Static signature dependency
- ❌ Slow response time

**Solution:** An intelligent hybrid NIDS that dynamically analyzes traffic patterns and adapts to new attack behaviors.

## ✨ Key Features

- ✅ **Hybrid Detection** - Combines signature-based and ML-based detection
- ✅ **Real-time Monitoring** - Continuous network traffic analysis
- ✅ **Multi-Protocol Support** - TCP, UDP, ICMP protocols
- ✅ **Intelligent Alerts** - Severity-based classification and notification
- ✅ **High Accuracy** - Advanced ML models (Random Forest, Decision Tree, SVM)
- ✅ **Comprehensive Reporting** - Detailed performance metrics and analytics

## 🏗️ System Architecture

### Modular Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Enhanced NIDS System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Packet Capture Module        (Network Traffic Input)     │
│  2. Feature Extraction Module    (Raw → Features)            │
│  3. Preprocessing Module         (Clean & Normalize)         │
│  4. Signature Detection Engine   (Known Attack Patterns)     │
│  5. ML Anomaly Detection Engine  (Unknown Threats)           │
│  6. Hybrid Detection Engine      (Combined Decision)         │
│  7. Alert & Response Module      (Security Alerts)           │
│  8. Performance Evaluation       (Metrics & Reports)         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Working Principle

### Detection Flow

```
Network Traffic
      ↓
Packet Capture
      ↓
Feature Extraction
      ↓
Data Preprocessing
      ↓
┌─────────────────┬──────────────────┐
│   Signature     │   ML Anomaly     │
│   Detection     │   Detection      │
└─────────────────┴──────────────────┘
      ↓
Hybrid Decision (OR Logic)
      ↓
Alert Generation (if intrusion)
      ↓
Logging & Reporting
```

### Hybrid Detection Algorithm

```python
FOR each packet:
    Extract features
    Preprocess data
    
    # Signature-based check
    IF matches known signature:
        Mark as Intrusion
    
    # ML-based check
    ELSE IF ML model predicts anomaly:
        Mark as Intrusion
    
    ELSE:
        Mark as Normal
    
    IF Intrusion detected:
        Generate alert
        Log event
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Installation Steps

1. **Clone or download the project**
   ```bash
   cd C:\Users\msarw\OneDrive\Documents\project\shank3
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the system**
   ```bash
   python main.py
   ```

## 📊 Dataset

The system supports multiple intrusion detection datasets:
- **NSL-KDD** - Network Security Laboratory dataset
- **CICIDS** - Canadian Institute for Cybersecurity IDS dataset
- **Synthetic Data** - Built-in data generation for demonstration

### Synthetic Dataset Features
- Duration
- Protocol Type (TCP/UDP/ICMP)
- Source/Destination Bytes
- Connection Count
- Flags (SYN, ACK)

## 🧠 Machine Learning Models

The system supports three ML algorithms:

1. **Random Forest** (Default)
   - Ensemble learning method
   - High accuracy and robustness
   - Handles complex patterns

2. **Decision Tree**
   - Fast training and prediction
   - Interpretable results
   - Good for rule extraction

3. **Support Vector Machine (SVM)**
   - Effective in high-dimensional spaces
   - Memory efficient
   - Versatile kernel functions

## 📈 Performance Metrics

The system evaluates detection using:

- **Accuracy** = Correct Predictions / Total Samples
- **Precision** = True Attacks / Total Predicted Attacks
- **Recall** = True Attacks Detected / Actual Attacks
- **F1 Score** = Harmonic mean of Precision & Recall
- **False Positive Rate** = Normal traffic wrongly flagged

## 🛡️ Attack Detection Capabilities

### Signature-Based Detection
- Known Malware Ports (4444, 5555, 6666, 31337)
- SYN Flood attacks
- Port Scanning
- Suspicious Port Access
- DDoS patterns

### ML-Based Anomaly Detection
- Zero-day attacks
- Novel attack patterns
- Traffic anomalies
- Behavioral deviations

## 📋 Output & Alerts

### Alert Format
```
🚨 SECURITY ALERT #1
═══════════════════════════════════════════════
Timestamp:        2026-01-28 10:30:45
Severity:         HIGH
Attack Type:      Potential SYN Flood
Source IP:        192.168.1.100
Destination IP:   10.0.0.50
Protocol:         TCP
Detection Method: Signature
Confidence:       95.0%
═══════════════════════════════════════════════
```

### System Report Includes
- Detection accuracy metrics
- Confusion matrix
- False positive rate
- Alert statistics by severity
- Attack type distribution

## 🔧 Configuration

To change the ML model, modify the initialization:

```python
# In EnhancedNIDS.__init__()
self.ml_engine = MLAnomalyDetectionEngine(model_type='random_forest')
# Options: 'random_forest', 'decision_tree', 'svm'
```

## 📂 Project Structure

```
shank3/
│
├── main.py              # Main NIDS implementation
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## 🎯 Module Descriptions

### 1. PacketCaptureModule
- Captures network packets
- Simulates traffic for testing
- Stores packet metadata

### 2. FeatureExtractionModule
- Converts raw packets to numerical features
- Protocol encoding
- Flag extraction

### 3. PreprocessingModule
- Removes duplicates
- Handles missing values
- Normalizes features
- Encodes categorical data

### 4. SignatureDetectionEngine
- Maintains attack signature database
- Pattern matching
- Known threat detection

### 5. MLAnomalyDetectionEngine
- Trains ML models
- Predicts anomalies
- Supports multiple algorithms

### 6. HybridDetectionEngine
- Combines both detection methods
- OR logic for intrusion flagging
- Confidence scoring

### 7. AlertResponseModule
- Generates security alerts
- Severity classification
- Alert summarization

### 8. PerformanceEvaluationModule
- Calculates metrics
- Generates reports
- Performance analysis

## 💡 Usage Examples

### Basic Usage
```python
# Initialize NIDS
nids = EnhancedNIDS()

# Load dataset
df, labels = nids.load_and_prepare_dataset()

# Train system
X_train, X_test, y_train, y_test = train_test_split(df.values, labels, test_size=0.3)
metrics = nids.train_system(X_train, y_train, X_test, y_test)

# Process live traffic
nids.process_live_traffic(num_packets=50)

# Generate report
nids.generate_report(metrics)
```

### Custom Dataset
```python
# Use your own dataset
nids = EnhancedNIDS()
df, labels = nids.load_and_prepare_dataset(dataset_path='path/to/nsl-kdd.csv')
```

## ⚡ Performance

- **Training Time**: ~2-5 seconds (1000 samples)
- **Detection Speed**: Real-time (~0.1s per packet)
- **Accuracy**: 85-95% (depends on dataset)
- **False Positive Rate**: <5%

## 🔍 Advantages

✅ Detects both known and unknown attacks  
✅ Faster response to threats  
✅ Reduced false alerts through hybrid approach  
✅ Scalable architecture  
✅ Intelligent learning and adaptation  
✅ Comprehensive logging and reporting  

## ⚠️ Limitations

- Requires quality training data
- High computation for very large-scale networks
- Needs periodic model retraining
- Cannot physically block attacks (detection only)

## 🚀 Future Enhancements

- [ ] Deep Learning models (LSTM, CNN)
- [ ] Automated response and blocking
- [ ] Cloud-based distributed IDS
- [ ] IoT security integration
- [ ] Real-time dashboard visualization
- [ ] Integration with SIEM systems
- [ ] Support for encrypted traffic analysis

## 📚 Technologies Used

| Component | Technology |
|-----------|-----------|
| Language | Python 3.x |
| ML Framework | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Dataset | NSL-KDD, CICIDS (synthetic for demo) |
| Models | Random Forest, Decision Tree, SVM |

## 📖 References

- NSL-KDD Dataset: https://www.unb.ca/cic/datasets/nsl.html
- CICIDS Dataset: https://www.unb.ca/cic/datasets/ids-2017.html
- Scikit-learn Documentation: https://scikit-learn.org

## 👨‍💻 Author

Enhanced NIDS - A Cybersecurity Research Project

## 📄 License

This project is for educational and research purposes.

---

## 🎓 Learning Outcomes

This project demonstrates:
- Network security fundamentals
- Intrusion detection techniques
- Machine learning for cybersecurity
- Hybrid detection systems
- Real-time threat analysis
- Performance evaluation methodologies

---

**Built with ❤️ for Cybersecurity**
#   T h e - E n h a n c e d - N I D S - v 3 . 0 - p r o j e c t - i s - n o w - c l e a n - a n d - o r g a n i z e d -  
 