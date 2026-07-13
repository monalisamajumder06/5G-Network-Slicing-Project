import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("embb_dataset_v2_with_target.csv")

# Features and target
features = df[["throughput_mbps", "active_users", "packet_loss"]].values
target = df["future_throughput"].values

# Sliding window
sequence_length = 10

X = []
y = []

for i in range(len(features) - sequence_length):
    X.append(features[i:i + sequence_length])
    y.append(target[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# Feature scaler
feature_scaler = MinMaxScaler()

# Reshape for scaling
X_train_reshaped = X_train.reshape(-1, 3)
X_test_reshaped = X_test.reshape(-1, 3)

X_train_scaled = feature_scaler.fit_transform(X_train_reshaped)
X_test_scaled = feature_scaler.transform(X_test_reshaped)

# Restore original shape
X_train_scaled = X_train_scaled.reshape(
    X_train.shape
)

X_test_scaled = X_test_scaled.reshape(
    X_test.shape
)

print("Original:")
print(X_train[0])

print("\nScaled:")
print(X_train_scaled[0])
