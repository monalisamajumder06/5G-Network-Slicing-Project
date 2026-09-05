AI-Driven 5G Network Slicing with Time-Series Forecasting and Dynamic Resource Allocation

An end-to-end 5G Network Slicing project combining Open5GS, UERANSIM, time-series forecasting, and QoS-aware dynamic resource allocation.

Overview

5G Network Slicing allows a physical 5G network to be divided into logical networks with different performance requirements.

This project explores how machine learning can make network slicing more proactive by predicting future traffic and network conditions for three major 5G service categories:

eMBB — Enhanced Mobile Broadband
URLLC — Ultra-Reliable Low-Latency Communication
mMTC — Massive Machine-Type Communication

The project implements an end-to-end pipeline:

5G Testbed
    │
    ▼
Open5GS + UERANSIM
    │
    ▼
Traffic Generation & Data Collection
    │
    ▼
Preprocessing
    │
    ▼
Time-Series Forecasting
    │
    ├── eMBB  → Throughput Prediction
    ├── URLLC → Latency Prediction
    └── mMTC  → Packet-Rate Prediction
    │
    ▼
Predicted Network Demand
    │
    ▼
QoS-Aware Dynamic PRB Allocation
    │
    ▼
100 PRBs Distributed Across Slices

The final stage uses model predictions to dynamically distribute available Physical Resource Blocks (PRBs) between the three slices while respecting minimum resource guarantees.

Key Objectives
Build a practical 5G network slicing testbed.
Generate and process network traffic data.
Model traffic and QoS behavior for eMBB, URLLC, and mMTC.
Compare different time-series forecasting approaches.
Evaluate forecasting models using MAE and RMSE.
Use model predictions to drive dynamic resource allocation.
Compare dynamic allocation against a static baseline.
Explore prediction-driven and proactive 5G network management.
System Architecture

The project is organized into four major stages.

1. 5G Network Testbed

The networking environment is based on:

Open5GS — 5G Core Network
UERANSIM — UE and gNB simulator
Ubuntu Linux

Different UE loads are used to represent changing network conditions.

Experiments include:

1 UE
2 UEs
4 UEs
6 UEs
8 UEs
10 UEs
2. Dataset Generation and Preprocessing

The project uses network measurements and time-series datasets for the three network slices.

The datasets contain network measurements relevant to each slice and corresponding future prediction targets.

eMBB

Inputs:

Throughput
Packet Loss
Active Users

Target:

Future Throughput
URLLC

Inputs:

Latency
Jitter
Packet Loss
Active Users

Target:

Future Latency
mMTC

Inputs:

Packet Rate
Active Users

Target:

Future Packet Rate
Network Slices
eMBB — Enhanced Mobile Broadband

eMBB represents applications requiring high data rates.

Examples include:

Video streaming
Cloud gaming
Virtual reality
Augmented reality

Prediction target: Future Throughput

URLLC — Ultra-Reliable Low-Latency Communication

URLLC represents applications where latency and reliability are critical.

Examples include:

Industrial automation
Robotics
Autonomous systems
Remote control applications

Prediction target: Future Latency

mMTC — Massive Machine-Type Communication

mMTC represents deployments containing large numbers of low-data-rate devices.

Examples include:

IoT deployments
Smart cities
Environmental monitoring
Smart agriculture

Prediction target: Future Packet Rate

Machine Learning Models

Several time-series forecasting approaches were implemented and evaluated.

LSTM

Long Short-Term Memory networks were used for sequential network prediction.

LSTM models learn temporal dependencies in network measurements and provide a deep-learning approach for the forecasting pipeline.

PatchTST

PatchTST is a Transformer-based time-series forecasting architecture that processes temporal sequences using patches and self-attention.

It was evaluated for network prediction tasks, including the mMTC and URLLC experiments.

TimesFM

TimesFM is a pretrained foundation model for time-series forecasting.

It was evaluated as a pretrained forecasting approach and was also used with additional network covariates for the URLLC forecasting task.

Model Evaluation

Forecasting models are evaluated using:

MAE — Mean Absolute Error
RMSE — Root Mean Square Error

A persistence baseline is also used where applicable to determine whether a forecasting model provides meaningful improvement over simply using a previous observation as the prediction.

Selected Model Results
Slice	Selected Forecasting Approach
eMBB	LSTM
URLLC	TimesFM
mMTC	PatchTST

The repository contains the corresponding training and evaluation scripts.

Dynamic Resource Allocation

The forecasting stage is connected to a QoS-aware resource controller.

Instead of allocating a fixed number of PRBs to every slice, the controller converts predicted network conditions into normalized demand values.

Model Predictions
       │
       ▼
Demand Estimation
       │
       ▼
QoS-Aware Allocation
       │
       ▼
eMBB + URLLC + mMTC
       │
       ▼
Exactly 100 PRBs

The controller maintains minimum PRB guarantees:

eMBB  ≥ 20 PRBs
URLLC ≥ 30 PRBs
mMTC  ≥ 20 PRBs

The remaining PRBs are distributed according to the predicted demand of each slice.

This creates a prediction-driven resource allocation pipeline:

Predict
   ↓
Estimate Demand
   ↓
Allocate Resources
   ↓
Evaluate
Resource Allocation Results

The final controller was evaluated against a static allocation baseline.

Static Baseline
eMBB  = 33 PRBs
URLLC = 34 PRBs
mMTC  = 33 PRBs
Dynamic Controller

The controller generates varying allocations according to predicted network conditions while maintaining the total PRB constraint and minimum resource guarantees.

Validation
Exactly 100 PRBs allocated: Yes
Minimum PRB constraints respected: Yes
Unique allocations observed: 46
Alignment Evaluation
Approach	Alignment Error
Static Baseline	0.1481
Dynamic Controller	0.0852

Dynamic controller improvement: 42.44%

Lower alignment error indicates that the dynamic allocation follows the predicted resource demand more closely than the static allocation.

Resource Allocation Visualizations
Dynamic PRB Allocation Under Different UE Loads

QoS Pressure vs PRB Allocation

Dynamic Controller vs Static Baseline

Repository Structure
5G-Network-Slicing-Project/
│
├── datasets/
│   ├── embb/
│   ├── urllc/
│   └── mmtc/
│
├── plots/
│   ├── embb/
│   └── urllc/
│
├── results/
│   └── embb/
│
├── resource_controller/
│   ├── src/
│   │   └── controller.py
│   │
│   ├── plots/
│   │   ├── allocation_by_active_users.png
│   │   ├── dynamic_vs_static_alignment.png
│   │   └── pressure_vs_prb_allocation.png
│   │
│   └── results/
│       └── resource_allocation_results.csv
│
├── scripts/
│   ├── data_generation/
│   ├── embb/
│   ├── evaluation/
│   ├── preprocessing/
│   └── training/
│
├── .gitignore
└── README.md
Technologies
Category	Technology
Programming Language	Python
5G Core	Open5GS
UE/gNB Simulator	UERANSIM
Deep Learning	PyTorch
Time-Series Foundation Model	TimesFM
Data Processing	Pandas, NumPy
Visualization	Matplotlib
Evaluation	Scikit-learn
Operating System	Ubuntu Linux
Installation

Clone the repository:

git clone https://github.com/monalisamajumder06/5G-Network-Slicing-Project.git
cd 5G-Network-Slicing-Project

Install the required Python packages according to your environment.

The 5G testbed additionally requires Open5GS and UERANSIM to be configured separately.

Running the Project

The project can be executed as a pipeline.

Step 1 — Prepare the 5G Testbed

Configure:

Open5GS
UERANSIM
Subscribers
Network slices
UEs and gNB
Step 2 — Generate or Collect Network Data

Relevant scripts are available under:

scripts/data_generation/
scripts/embb/
Step 3 — Preprocess the Datasets

Preprocessing utilities are located under:

scripts/preprocessing/
Step 4 — Train Forecasting Models

Training scripts are located under:

scripts/training/
Step 5 — Evaluate Models

Evaluation scripts are located under:

scripts/evaluation/
Step 6 — Run Resource Allocation

The final resource controller is located at:

resource_controller/src/controller.py

It consumes prediction data and generates dynamic PRB allocations for the three network slices.

End-to-End Workflow
                 ┌───────────────────────┐
                 │    Open5GS + UERANSIM │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ Traffic & QoS Data    │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │    Preprocessing      │
                 └───────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │    Time-Series Forecasting   │
              │                              │
              │  LSTM / PatchTST / TimesFM  │
              └──────────────┬───────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Predicted Conditions │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ QoS-Aware Controller │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Dynamic PRB Allocation│
                  └──────────┬───────────┘
                             │
                             ▼
                       100 PRBs / Cell
Limitations

The current resource controller is a rule-based QoS-aware allocation mechanism, rather than a reinforcement-learning or optimization-based network orchestrator.

The current evaluation therefore demonstrates the connection between:

predicted network demand → dynamic resource allocation

rather than claiming a fully autonomous 5G network orchestrator.

The current controller provides a foundation that can be extended with more advanced optimization and learning-based resource allocation methods.

Future Work

Potential extensions include:

Reinforcement Learning-based slice orchestration
Optimization-based PRB allocation
Real-time closed-loop resource allocation
Integration with live 5G network metrics
Real-time model inference
Physical 5G hardware deployment
SDN integration
Automated SLA/QoS enforcement
Closed-loop Zero-Touch Network Slicing
References
Open5GS
UERANSIM
PyTorch
TimesFM
PatchTST
3GPP 5G Network Slicing specifications
Author

Monalisa Majumder

B.Tech Computer Science and Engineering
SRM Institute of Science and Technology

Research Internship under Prof. Soumya Kanti Ghosh
Department of Computer Science and Engineering
Indian Institute of Technology Kharagpur
