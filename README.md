# Smart Helmet – Machine Learning (Anomaly Detection)

This repository contains the **machine learning part** of the Smart Helmet project.  
The goal is to detect **abnormal riding events** (such as crashes or dangerous impacts) using sensor data from the helmet.

The approach is based on **unsupervised anomaly detection**, where the model learns what *normal riding* looks like and flags unusual patterns.

---

## 📁 Project Structure

.
├── driving_clean.csv
├── smart_helmet_anomaly_model_report.ipynb
├── smart_helmet_algorithm_comparison_step_by_step.ipynb
├── train_anomaly_model.py
├── README.md

### Files explained

- **driving_clean.csv**  
  Cleaned telemetry data (IMU, speed, timestamps) used for training and evaluation.

- **smart_helmet_anomaly_model_report.ipynb**  
  Main notebook that explains:
  - feature engineering (rolling windows)
  - training Isolation Forest
  - threshold selection
  - anomaly visualization  

- **smart_helmet_algorithm_comparison_step_by_step.ipynb**  
  Each algorithm is evaluated separately (Isolation Forest, One-Class SVM, LOF),
  followed by a final comparison showing why Isolation Forest was selected.

- **train_anomaly_model.py**  
  Script version of the training pipeline for reproducibility and backend use.

---

## 🧠 Model Overview

- Rolling-window feature extraction
- Unsupervised anomaly detection
- Percentile-based thresholding

### Algorithms Tested
- Isolation Forest (final choice)
- One-Class SVM
- Local Outlier Factor (novelty)

---

## 📊 Evaluation Strategy

Because real crash labels are not available:
- No traditional accuracy is used
- Models are compared using:
  - anomaly rate on test data
  - stability over time
  - runtime performance

---

## 🚀 How to Run

```bash
conda create -n helmet_ml python=3.11 -y
conda activate helmet_ml
conda install numpy pandas matplotlib scikit-learn joblib -y
jupyter notebook
```

---

## 🔗 Integration

The trained model is designed to be integrated into the Smart Helmet backend
for real-time crash and anomaly detection.

---

Smart Helmet Project – Machine Learning Module
