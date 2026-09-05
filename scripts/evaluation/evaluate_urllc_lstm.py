import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("urllc_dataset_with_target.csv")

features = df[["latency_ms", "jitter_ms", "packet_loss", "active_users"]].values
target = df["future_latency_ms"].values

# -----------------------------
# Sliding Window
# -----------------------------
sequence_length = 5

X = []
y = []

for i in range(len(features) - sequence_length):
    X.append(features[i:i + sequence_length])
    y.append(target[i + sequence_length])

X = np.array(X)
y = np.array(y)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# -----------------------------
# Normalization
# -----------------------------
scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train_reshaped = X_train.reshape(-1, 4)
X_test_reshaped = X_test.reshape(-1, 4)

X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

X_train_scaled = X_train_scaled.reshape(X_train.shape)
X_test_scaled = X_test_scaled.reshape(X_test.shape)

target_scaler.fit(y_train.reshape(-1, 1))

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

# -----------------------------
# LSTM Model
# -----------------------------
class LSTMModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(128, 1)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        x = hidden[-1]

        x = self.fc(x)

        return x

# -----------------------------
# Load Saved Model
# -----------------------------
model = LSTMModel()

model.load_state_dict(
    torch.load(
        "urllc_lstm.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

print("Model loaded successfully.")

# -----------------------------
# Baseline
# -----------------------------
baseline_pred = X_test[:, -1, 0]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_pred
    )
)

print("\nBaseline Results")
print(f"MAE  : {baseline_mae:.4f}")
print(f"RMSE : {baseline_rmse:.4f}")

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():

    y_pred_scaled = model(X_test_tensor).numpy()

y_pred = target_scaler.inverse_transform(y_pred_scaled)

y_pred = np.clip(y_pred, 0, None)

# -----------------------------
# Metrics
# -----------------------------
mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

print("\nLSTM Results")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# -----------------------------
# Prediction Plot
# -----------------------------
plt.figure(figsize=(12,6))

plt.plot(
    y_test[:200],
    label="Actual"
)

plt.plot(
    y_pred[:200],
    label="Predicted"
)

plt.xlabel("Sample")
plt.ylabel("Latency (ms)")
plt.title("Actual vs Predicted Latency")

plt.legend()
plt.grid(True)

plt.show()

# -----------------------------
# Residual Plot
# -----------------------------
errors = y_test - y_pred.flatten()

plt.figure(figsize=(12,5))

plt.plot(errors[:200])

plt.title("Prediction Error")
plt.xlabel("Sample")
plt.ylabel("Error (ms)")

plt.grid(True)

plt.show()

# -----------------------------
# Scatter Plot
# -----------------------------
plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.title("Actual vs Predicted Scatter")

plt.grid(True)

plt.show()
