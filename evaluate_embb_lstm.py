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
df = pd.read_csv("embb_dataset_v4_with_target.csv")

features = df[
    [
        "throughput_mbps",
        "active_users",
        "packet_loss"
    ]
].values

target = df["future_throughput"].values

# -----------------------------
# Sliding Window
# -----------------------------
sequence_length = 30

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
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train_scaled = feature_scaler.fit_transform(
    X_train.reshape(-1,3)
).reshape(X_train.shape)

X_test_scaled = feature_scaler.transform(
    X_test.reshape(-1,3)
).reshape(X_test.shape)

target_scaler.fit(
    y_train.reshape(-1,1)
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

# -----------------------------
# Model Definition
# -----------------------------
class LSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(
            128,
            1
        )

    def forward(self,x):

        output,(hidden,cell)=self.lstm(x)

        out=self.fc(hidden[-1])

        return out

# -----------------------------
# Load Saved Model
# -----------------------------
model = LSTMModel()

model.load_state_dict(
    torch.load(
        "embb_lstm_v4.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

print("Model loaded successfully.")

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():

    y_pred_scaled = model(
        X_test_tensor
    ).numpy()

y_pred = target_scaler.inverse_transform(
    y_pred_scaled
).flatten()

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

print("\nModel Performance")
print("-------------------------")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

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

print("\nBaseline Performance")
print("-------------------------")
print(f"MAE  : {baseline_mae:.4f}")
print(f"RMSE : {baseline_rmse:.4f}")

# -----------------------------
# Actual vs Predicted
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

plt.title("Actual vs Predicted Throughput")

plt.xlabel("Sample")

plt.ylabel("Throughput (Mbps)")

plt.legend()

plt.grid(True)

plt.tight_layout()

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

plt.tight_layout()

plt.show()

# -----------------------------
# Residual Plot
# -----------------------------
residuals = y_test - y_pred

plt.figure(figsize=(12,5))

plt.plot(residuals)

plt.axhline(
    0,
    color='red',
    linestyle='--'
)

plt.xlabel("Sample")

plt.ylabel("Residual")

plt.title("Prediction Residuals")

plt.grid(True)

plt.tight_layout()

plt.show()

# -----------------------------
# Residual Histogram
# -----------------------------
plt.figure(figsize=(8,5))

plt.hist(
    residuals,
    bins=30
)

plt.xlabel("Residual")

plt.ylabel("Frequency")

plt.title("Residual Distribution")

plt.grid(True)

plt.tight_layout()

plt.show()
