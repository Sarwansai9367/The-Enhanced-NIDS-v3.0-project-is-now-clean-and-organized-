# Enhanced NIDS - Visual System Diagrams

## 📊 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│                  ENHANCED NETWORK INTRUSION DETECTION SYSTEM              │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  INPUT LAYER                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
│  │   TCP       │  │    UDP      │  │   ICMP      │                      │
│  │   Traffic   │  │   Traffic   │  │   Traffic   │                      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                      │
│         └─────────────────┴─────────────────┘                            │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  CAPTURE LAYER                                                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  PacketCaptureModule                                              │   │
│  │  • Timestamp recording                                            │   │
│  │  • IP address extraction                                          │   │
│  │  • Protocol identification                                        │   │
│  │  • Port number capture                                            │   │
│  │  • Flag parsing                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING LAYER                                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  FeatureExtractionModule                                          │   │
│  │  • Protocol encoding (TCP=0, UDP=1, ICMP=2)                       │   │
│  │  • Port number features                                           │   │
│  │  • Packet size features                                           │   │
│  │  • Flag features (SYN, ACK)                                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PREPROCESSING LAYER                                                      │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  PreprocessingModule                                              │   │
│  │  • Remove duplicates                                              │   │
│  │  • Handle missing values                                          │   │
│  │  • Normalize features (StandardScaler)                            │   │
│  │  • Encode categorical variables                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ├────────────────────────┐
                                  │                        │
                                  ▼                        ▼
┌───────────────────────────────────────┐  ┌──────────────────────────────┐
│  SIGNATURE DETECTION LAYER            │  │  ML DETECTION LAYER          │
│  ┌────────────────────────────────┐   │  │  ┌────────────────────────┐  │
│  │ SignatureDetectionEngine       │   │  │  │ MLAnomalyDetectionEngine│ │
│  │                                │   │  │  │                         │ │
│  │ Known Attack Patterns:         │   │  │  │ ML Models:              │ │
│  │ • Malware Ports                │   │  │  │ • Random Forest (100)   │ │
│  │   (4444, 5555, 6666, 31337)    │   │  │  │ • Decision Tree         │ │
│  │ • SYN Flood Detection          │   │  │  │ • SVM (RBF kernel)      │ │
│  │ • Port Scan Indicators         │   │  │  │                         │ │
│  │ • Suspicious Port Access       │   │  │  │ Output:                 │ │
│  │                                │   │  │  │ 0 = Normal              │ │
│  │ Output:                        │   │  │  │ 1 = Intrusion           │ │
│  │ (is_attack, attack_type)       │   │  │  │                         │ │
│  └────────────────────────────────┘   │  │  └────────────────────────┘  │
└───────────────┬───────────────────────┘  └──────────────┬───────────────┘
                │                                          │
                └──────────────────┬───────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  HYBRID DECISION LAYER                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  HybridDetectionEngine                                            │   │
│  │                                                                    │   │
│  │  Decision Logic (OR):                                             │   │
│  │  IF Signature Detection = ATTACK OR ML Detection = ATTACK         │   │
│  │  THEN Result = INTRUSION                                          │   │
│  │                                                                    │   │
│  │  Confidence Scoring:                                              │   │
│  │  • Signature only: 95%                                            │   │
│  │  • ML only: 85%                                                   │   │
│  │  • Both detect: 95%                                               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ALERT GENERATION LAYER                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  AlertResponseModule                                              │   │
│  │                                                                    │   │
│  │  Alert Structure:                                                 │   │
│  │  ┌──────────────────────────────────────────────────────────┐    │   │
│  │  │ 🚨 SECURITY ALERT #X                                      │    │   │
│  │  │ Timestamp:        [DateTime]                              │    │   │
│  │  │ Severity:         HIGH / MEDIUM / LOW                     │    │   │
│  │  │ Attack Type:      [Type]                                  │    │   │
│  │  │ Source IP:        [IP Address]                            │    │   │
│  │  │ Destination IP:   [IP Address]                            │    │   │
│  │  │ Protocol:         TCP / UDP / ICMP                        │    │   │
│  │  │ Detection Method: Signature / ML / Both                   │    │   │
│  │  │ Confidence:       [Percentage]                            │    │   │
│  │  └──────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  REPORTING & ANALYTICS LAYER                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  PerformanceEvaluationModule                                      │   │
│  │                                                                    │   │
│  │  Performance Metrics:                                             │   │
│  │  • Accuracy = (TP + TN) / Total                                   │   │
│  │  • Precision = TP / (TP + FP)                                     │   │
│  │  • Recall = TP / (TP + FN)                                        │   │
│  │  • F1 Score = 2 × (P × R) / (P + R)                               │   │
│  │  • False Positive Rate = FP / (FP + TN)                           │   │
│  │                                                                    │   │
│  │  Reports Generated:                                               │   │
│  │  • Confusion Matrix                                               │   │
│  │  • Classification Report                                          │   │
│  │  • Alert Summary Statistics                                       │   │
│  │  • Attack Type Distribution                                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Sequence

```
┌──────────┐
│ PACKET   │
│ RECEIVED │
└────┬─────┘
     │
     ▼
┌────────────────────┐
│ 1. CAPTURE         │  ◄── Timestamp, IPs, Ports, Protocol, Flags
│    PacketCapture   │
└────┬───────────────┘
     │
     ▼
┌────────────────────┐
│ 2. EXTRACT         │  ◄── Convert to numerical features
│    Features        │      protocol_type, packet_size, ports, flags
└────┬───────────────┘
     │
     ▼
┌────────────────────┐
│ 3. PREPROCESS      │  ◄── Normalize, scale, encode
│    Data            │      StandardScaler transformation
└────┬───────────────┘
     │
     ├─────────────────────┬─────────────────────┐
     │                     │                     │
     ▼                     ▼                     ▼
┌─────────┐          ┌─────────┐          ┌──────────┐
│ Sig Det │          │ ML Det  │          │ Hybrid   │
│ Engine  │          │ Engine  │          │ Engine   │
└────┬────┘          └────┬────┘          └────┬─────┘
     │                    │                     │
     │ Known patterns     │ Anomaly detection   │ Combined
     │ (95% conf)         │ (85% conf)          │ decision
     │                    │                     │
     └────────────────────┴─────────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ INTRUSION?    │
                  └───┬───────┬───┘
                      │       │
                  YES │       │ NO
                      │       │
                      ▼       ▼
              ┌────────┐   ┌────────┐
              │ ALERT  │   │ NORMAL │
              │ Generate│  │ Log    │
              └───┬────┘   └────────┘
                  │
                  ▼
          ┌──────────────┐
          │ REPORT       │
          │ Metrics      │
          └──────────────┘
```

---

## 🧠 Machine Learning Model Architecture

### Random Forest Classifier (Default)

```
┌─────────────────────────────────────────────────────────────────┐
│                    RANDOM FOREST CLASSIFIER                      │
│                     (100 Decision Trees)                         │
└─────────────────────────────────────────────────────────────────┘

Input: Normalized Feature Vector [8 features]
  │
  ├──► Tree 1  ──┐
  ├──► Tree 2  ──┤
  ├──► Tree 3  ──┤
  ├──► Tree 4  ──┤
  ├──► ...     ──┤
  ├──► Tree 98 ──┤
  ├──► Tree 99 ──┤
  └──► Tree 100 ─┤
                 │
                 ├──► Voting Mechanism
                 │    (Majority Vote)
                 │
                 ▼
         ┌──────────────┐
         │  Prediction  │
         │  0 = Normal  │
         │  1 = Attack  │
         └──────────────┘
```

---

## 📊 Detection Decision Matrix

```
┌────────────────────────────────────────────────────────────────────┐
│                    HYBRID DETECTION MATRIX                          │
└────────────────────────────────────────────────────────────────────┘

                           ML Detection
                     │  Normal  │  Intrusion  │
  ───────────────────┼──────────┼─────────────┤
  Signature  Normal  │  NORMAL  │  INTRUSION  │
  Detection          │  (99%)   │    (85%)    │
             ────────┼──────────┼─────────────┤
             Attack  │ INTRUSION│  INTRUSION  │
                     │  (95%)   │    (95%)    │
  ───────────────────┴──────────┴─────────────┘

Legend:
• NORMAL: No threat detected
• INTRUSION: Threat detected
• (%) = Confidence level
```

---

## 🎯 Alert Severity Classification

```
┌──────────────────────────────────────────────────────────────┐
│                   SEVERITY CLASSIFICATION                     │
└──────────────────────────────────────────────────────────────┘

Attack Type                              Severity Level
────────────────────────────────────────────────────────
DDoS Attack                          ──►  🔴 HIGH
SYN Flood                            ──►  🔴 HIGH
Known Malware Port                   ──►  🔴 HIGH
                                          
Port Scan                            ──►  🟡 MEDIUM
Suspicious Port Access               ──►  🟡 MEDIUM
                                          
Anomaly Detected                     ──►  🟢 LOW
Other Patterns                       ──►  🟢 LOW
```

---

## 📈 Performance Metrics Visualization

```
┌────────────────────────────────────────────────────────────────┐
│                    CONFUSION MATRIX                             │
└────────────────────────────────────────────────────────────────┘

                        Predicted
                  │  Normal  │  Attack  │
  ────────────────┼──────────┼──────────┤
  Actual  Normal  │    TN    │    FP    │
          ────────┼──────────┼──────────┤
          Attack  │    FN    │    TP    │
  ────────────────┴──────────┴──────────┘

  TN = True Negative  (Correctly identified normal)
  TP = True Positive  (Correctly identified attack)
  FN = False Negative (Missed attack)
  FP = False Positive (False alarm)

┌────────────────────────────────────────────────────────────────┐
│                    METRICS CALCULATION                          │
└────────────────────────────────────────────────────────────────┘

  Accuracy  = (TP + TN) / (TP + TN + FP + FN)
  
  Precision = TP / (TP + FP)
  
  Recall    = TP / (TP + FN)
  
  F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)
  
  FPR       = FP / (FP + TN)
```

---

## 🔄 System State Diagram

```
┌──────────┐
│  INIT    │  ◄── System starts
└────┬─────┘
     │
     ▼
┌──────────┐
│  LOAD    │  ◄── Load dataset or capture traffic
└────┬─────┘
     │
     ▼
┌──────────┐
│  TRAIN   │  ◄── Train ML model
└────┬─────┘
     │
     ▼
┌──────────┐
│  READY   │  ◄── System ready for detection
└────┬─────┘
     │
     ├──────────────┐
     │              │
     ▼              ▼
┌──────────┐   ┌──────────┐
│ DETECT   │   │ MONITOR  │
└────┬─────┘   └────┬─────┘
     │              │
     ▼              ▼
┌──────────┐   ┌──────────┐
│ ALERT    │   │  LOG     │
└────┬─────┘   └────┬─────┘
     │              │
     └──────┬───────┘
            │
            ▼
       ┌──────────┐
       │ REPORT   │
       └──────────┘
```

---

## 🛡️ Security Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                  THREAT DETECTION WORKFLOW                       │
└─────────────────────────────────────────────────────────────────┘

Normal Traffic Flow:
───────────────────
Packet ──► Capture ──► Extract ──► Detect ──► Normal ──► Log


Intrusion Detection Flow:
─────────────────────────
Packet ──► Capture ──► Extract ──► Detect ──► Intrusion ──┐
                                                           │
                                                           ▼
                                               ┌─────────────────┐
                                               │ Alert Generated │
                                               └────────┬────────┘
                                                        │
                     ┌──────────────────────────────────┴─┐
                     │                                    │
                     ▼                                    ▼
              ┌─────────────┐                   ┌─────────────┐
              │  Log Alert  │                   │ Display To  │
              │  To Storage │                   │  Admin      │
              └─────────────┘                   └─────────────┘
```

---

## 📊 Example Test Results

```
┌─────────────────────────────────────────────────────────────────┐
│               TEST RUN RESULTS (20 PACKETS)                      │
└─────────────────────────────────────────────────────────────────┘

Packet #  │  Protocol  │  Detection  │  Severity  │  Type
──────────┼────────────┼─────────────┼────────────┼──────────────
    1     │    UDP     │   Anomaly   │    LOW     │  ML
    2     │    TCP     │   Anomaly   │    LOW     │  ML
    3     │    TCP     │   Anomaly   │    LOW     │  ML
    4     │    UDP     │   Anomaly   │    LOW     │  ML
    5     │    ICMP    │  SYN Flood  │   HIGH     │  Signature
    6     │    TCP     │   Normal    │     -      │  -
    7     │    TCP     │   Anomaly   │    LOW     │  ML
    8     │    UDP     │  Suspicious │   MEDIUM   │  Signature+ML
    9     │    TCP     │   Normal    │     -      │  -
   10     │    TCP     │   Anomaly   │    LOW     │  ML
   ...    │    ...     │     ...     │    ...     │  ...

───────────────────────────────────────────────────────────────────
SUMMARY:
  • Total Packets: 20
  • Intrusions: 14 (70%)
  • Normal: 6 (30%)
  • High Severity: 1
  • Medium Severity: 2
  • Low Severity: 11
───────────────────────────────────────────────────────────────────
```

---

**Visual Documentation Complete**  
**Last Updated**: January 28, 2026
