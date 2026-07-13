# AI-Driven Traffic Prediction for Zero-Touch 5G Network Slicing

## Overview

This project presents an AI-driven framework for proactive traffic prediction in 5G Network Slicing. The proposed system predicts future network traffic demand for the three standardized 5G network slices—Enhanced Mobile Broadband (eMBB), Ultra-Reliable Low-Latency Communication (URLLC), and Massive Machine-Type Communication (mMTC)—using state-of-the-art time-series forecasting models.

A complete end-to-end 5G testbed was developed using Open5GS and UERANSIM to generate both synthetic and real network traffic datasets. These datasets were used to train and evaluate Long Short-Term Memory (LSTM), PatchTST, and TimesFM models. The predicted traffic demand can be utilized for proactive resource allocation, enabling the vision of Zero-Touch Network Slicing.

---

# Objectives

- Build a complete Open5GS-UERANSIM based 5G testbed.
- Generate synthetic and real datasets for all three network slices.
- Predict future network traffic using deep learning models.
- Compare multiple forecasting approaches.
- Enable proactive network resource allocation.
- Move towards intelligent Zero-Touch Network Slicing.

---

# System Architecture

The project consists of five major stages:

1. Deployment of a 5G Network Slicing testbed.
2. Traffic generation and dataset collection.
3. Data preprocessing.
4. Time-series traffic prediction.
5. Performance evaluation.

---

# Network Slices

## eMBB (Enhanced Mobile Broadband)

Prediction Target:
- Future Throughput

Applications:
- Video Streaming
- Cloud Gaming
- Virtual Reality
- Augmented Reality

---

## URLLC (Ultra-Reliable Low-Latency Communication)

Prediction Target:
- Future Latency

Applications:
- Autonomous Vehicles
- Industrial Automation
- Robotics
- Remote Healthcare

---

## mMTC (Massive Machine-Type Communication)

Prediction Target:
- Future Packet Rate

Applications:
- Smart Cities
- IoT
- Smart Agriculture
- Environmental Monitoring

---

# Technologies Used

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Operating System | Ubuntu |
| 5G Core | Open5GS |
| UE & gNB Simulator | UERANSIM |
| Database | MongoDB |
| Machine Learning | PyTorch |
| Foundation Model | TimesFM |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib |
| Evaluation | Scikit-learn |

---

#  Dataset Description

Both synthetic and real datasets were generated for each network slice.

## Synthetic Datasets

- eMBB Dataset
- URLLC Dataset
- mMTC Dataset

## Real Datasets

Traffic was collected from the deployed Open5GS-UERANSIM testbed using experiments with varying numbers of active users.

Experiments were conducted with:

- 1 UE
- 2 UEs
- 4 UEs
- 6 UEs
- 8 UEs
- 10 UEs

---

#  Features Used

## eMBB

- Throughput
- Packet Loss
- Active Users

Target:

- Future Throughput

---

## URLLC

- Latency
- Jitter
- Packet Loss
- Active Users

Target:

- Future Latency

---

## mMTC

- Packet Rate
- Active Users

Target:

- Future Packet Rate

---

# Machine Learning Models

## LSTM

A recurrent neural network capable of learning long-term temporal dependencies in sequential traffic data.

---

## PatchTST

A Transformer-based forecasting model that processes time-series data as patches using self-attention mechanisms.

---

## TimesFM

A pretrained foundation model for general-purpose time-series forecasting evaluated on the synthetic datasets.

---

# Performance Evaluation

Performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)

Each model was also compared against a Persistence Baseline.

---

# Summary of Results

### Synthetic Datasets

| Slice | Best Model |
|---------|------------|
| eMBB | LSTM |
| URLLC | TimesFM |
| mMTC | TimesFM |

---

### Real Datasets

| Slice | Best Model |
|---------|------------|
| eMBB | LSTM |
| URLLC | PatchTST |
| mMTC | PatchTST |

---

# Project Structure

```
5G-Network-Slicing-Project/

├── datasets/
├── embb_udp/
├── urllc_latency/
├── mmtc_udp/
├── scripts/
├── models/
├── preprocessing/
├── results/
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/monalisamajumder06/5G-Network-Slicing-Project.git
```

Move into the project directory

```bash
cd 5G-Network-Slicing-Project
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

1. Configure the Open5GS core.
2. Start MongoDB.
3. Launch Open5GS services.
4. Start the UERANSIM gNB.
5. Register User Equipments.
6. Generate traffic.
7. Train prediction models.
8. Evaluate forecasting performance.

---

# Future Work

- Dynamic resource allocation using predicted traffic.
- Reinforcement Learning-based slice orchestration.
- Real-time deployment on physical 5G hardware.
- Integration with Software Defined Networking (SDN).
- Closed-loop Zero-Touch Network Slicing.

---

# 👨‍💻 Author

**Monalisa Majumder**

B.Tech Computer Science and Engineering

SRM Institute of Science and Technology

Internship under **Prof. Soumya Kanti Ghosh**

Department of Computer Science and Engineering

Indian Institute of Technology Kharagpur

---

# References

- Open5GS
- UERANSIM
- TimesFM
- PatchTST
- PyTorch
- 3GPP 5G Standards

---

## If you found this project useful, consider giving it a star!
