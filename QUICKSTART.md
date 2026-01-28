# Enhanced NIDS - Quick Start Guide

## 🚀 Quick Installation & Execution

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the System
```bash
python main.py
```

## 📊 Expected Output

The system will execute in 4 phases:

### **PHASE 1: DATA PREPARATION**
- Creates or loads network traffic dataset
- Splits into training and testing sets
- Shows data distribution (Normal vs Intrusion)

### **PHASE 2: MODEL TRAINING**
- Trains Random Forest ML model
- Normalizes features
- Evaluates on test set
- Shows performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - False Positive Rate

### **PHASE 3: REAL-TIME DETECTION**
- Simulates live network traffic
- Processes packets through hybrid detection
- Generates real-time security alerts
- Shows:
  - Alert ID
  - Timestamp
  - Severity (HIGH/MEDIUM/LOW)
  - Attack Type
  - Source/Destination IPs
  - Protocol
  - Detection Method (Signature/ML/Both)
  - Confidence Level

### **PHASE 4: FINAL REPORT**
- Comprehensive system summary
- Overall detection metrics
- Alert statistics by severity
- Attack type distribution

## 🎯 Sample Output

```
======================================================================
🚨 SECURITY ALERT #1
======================================================================
Timestamp:        2026-01-28 07:51:07.860225
Severity:         HIGH
Attack Type:      Potential SYN Flood
Source IP:        192.168.1.23
Destination IP:   10.0.0.97
Protocol:         TCP
Detection Method: Signature
Confidence:       95.0%
======================================================================
```

## 🔧 Customization Options

### Change ML Model
Edit `main.py` line 444:
```python
self.ml_engine = MLAnomalyDetectionEngine(model_type='decision_tree')
# Options: 'random_forest', 'decision_tree', 'svm'
```

### Adjust Traffic Volume
Edit `main.py` line 637:
```python
nids.process_live_traffic(num_packets=50)  # Change from 20 to 50
```

### Use Real Dataset
```python
# Edit main() function
df, labels = nids.load_and_prepare_dataset(dataset_path='path/to/dataset.csv')
```

## 📈 Performance Benchmarks

With default settings (1000 samples):
- **Training Time**: ~0.2-0.5 seconds
- **Detection Speed**: ~0.1s per packet (real-time simulation)
- **Accuracy**: 85-95% (depending on dataset)
- **False Positive Rate**: <11%

## 🛡️ Detection Capabilities

### Signature-Based Detection
- ✓ Known Malware Ports (4444, 5555, 6666, 31337)
- ✓ SYN Flood attacks
- ✓ Suspicious port access patterns
- ✓ Port scanning indicators

### ML-Based Anomaly Detection
- ✓ Zero-day attacks
- ✓ Unknown threat patterns
- ✓ Behavioral anomalies
- ✓ Traffic deviations

### Hybrid Detection
- ✓ Combined signature + ML analysis
- ✓ Higher detection coverage
- ✓ Lower false negatives
- ✓ Confidence scoring

## 📝 Key Modules

1. **PacketCaptureModule** - Network packet capture
2. **FeatureExtractionModule** - Feature engineering
3. **PreprocessingModule** - Data cleaning and normalization
4. **SignatureDetectionEngine** - Pattern matching
5. **MLAnomalyDetectionEngine** - Machine learning detection
6. **HybridDetectionEngine** - Combined decision making
7. **AlertResponseModule** - Alert generation and management
8. **PerformanceEvaluationModule** - Metrics and reporting

## 🎓 Educational Use

This system demonstrates:
- Network intrusion detection principles
- Hybrid detection architectures
- Machine learning for cybersecurity
- Real-time threat analysis
- Security alert management
- Performance evaluation techniques

## ⚠️ Important Notes

- System uses synthetic data by default for demonstration
- For production use, integrate with real packet capture tools
- Requires quality training data for optimal performance
- Model should be retrained periodically with new attack patterns
- This is a detection system (not prevention/blocking)

## 🔍 Troubleshooting

### Issue: Import errors
**Solution**: Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Low accuracy
**Solution**: Use more training data or real dataset

### Issue: Too many false positives
**Solution**: Adjust detection thresholds or retrain with balanced dataset

## 📚 Next Steps

1. Test with real network datasets (NSL-KDD, CICIDS)
2. Integrate with network monitoring tools
3. Customize attack signatures
4. Tune ML model parameters
5. Add custom alert handlers
6. Implement logging to file/database

---

**System Status**: ✅ Fully Operational
**Last Updated**: January 28, 2026
is 