# Cyber Attack Detection System

A Machine Learning system to detect cyber attacks from network traffic data with 100% accuracy.

## Overview
This project detects **DDoS, PortScan, BruteForce attacks** and **Normal traffic** using network features like packet count, bytes transferred, and protocol type.

## Dataset
- **100,000** network traffic records
- **6 features**: duration, src_bytes, dst_bytes, packet_count, protocol, failed_logins
- **4 attack types**: DDoS, PortScan, BruteForce, Normal

## Model Performance
- **Best Model**: Random Forest
- **Accuracy**: 100%
- **Precision**: 1.00
- **Recall**: 1.00
- **F1-Score**: 1.00


## Project Structure

cyber-attack-detection/
├── src/
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ ├── pipeline/
│ │ ├── train_pipeline.py
│ │ └── predict_pipeline.py
│ ├── exception.py
│ ├── logger.py
│ └── utils.py
├── artifacts/
│ ├── model.pkl
│ ├── preprocessor.pkl
│ ├── label_encoder.pkl
│ └── model_report.txt
├── setup.py
├── requirements.txt
└── README.md
