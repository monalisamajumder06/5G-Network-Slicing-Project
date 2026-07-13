import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("embb_dataset_v2_with_target.csv")

# Input features
features = df[["throughput_mbps", "active_users", "packet_loss"]].values

# Target
target = df["future_throughput"].values

# Window size
sequence_length = 10

X = []
y = []

# Sliding window
for i in range(len(features) - sequence_length):

    X.append(features[i:i + sequence_length])

    y.append(target[i + sequence_length])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst sequence:\n")
print(X[0])

print("\nFirst target:")
print(y[0])
