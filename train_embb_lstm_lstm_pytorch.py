import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("embb_dataset_v4_with_target.csv")

# Features and target
features = df[
    ["throughput_mbps",
     "active_users",
     "packet_loss"]
].values

target = df["future_throughput"].values

# Sliding window
sequence_length = 30

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

# Normalization
scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train_reshaped = X_train.reshape(-1, 3)
X_test_reshaped = X_test.reshape(-1, 3)

X_train_scaled = scaler.fit_transform(
    X_train_reshaped
)

X_test_scaled = scaler.transform(
    X_test_reshaped
)

X_train_scaled = X_train_scaled.reshape(
    X_train.shape
)

X_test_scaled = X_test_scaled.reshape(
    X_test.shape
)

y_train_scaled = target_scaler.fit_transform(
    y_train.reshape(-1, 1)
)

y_test_scaled = target_scaler.transform(
    y_test.reshape(-1, 1)
)

# Convert to tensors
X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_scaled,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test_scaled,
    dtype=torch.float32
)

# DataLoader
train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False
)

# LSTM Model
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

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        out = self.fc(
            hidden[-1]
        )

        return out

model = LSTMModel()

# Loss Function
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005
)

# Training
epochs = 100

train_losses = []

for epoch in range(epochs):

    model.train()

    epoch_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    train_losses.append(avg_loss)

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {avg_loss:.4f}"
    )

# Plot Loss
plt.figure(figsize=(10,5))

plt.plot(train_losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.grid(True)

plt.show()

# Baseline
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
print("MAE:", baseline_mae)
print("RMSE:", baseline_rmse)

# Evaluation
model.eval()

with torch.no_grad():

    y_pred = model(
        X_test_tensor
    ).numpy()

# Back to original scale
y_pred = target_scaler.inverse_transform(
    y_pred
)

print("\nPrediction statistics:")
print("Min:", y_pred.min())
print("Max:", y_pred.max())
print("Mean:", y_pred.mean())

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

print("\nMAE:", mae)
print("RMSE:", rmse)

# Prediction Plot
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
plt.ylabel("Throughput")

plt.title(
    "Actual vs Predicted Throughput"
)

plt.legend()

plt.grid(True)

plt.show()

# Save Model
torch.save(
    model.state_dict(),
    "embb_lstm_v4.pth"
)

print(
    "\nModel saved as embb_lstm_v4.pth"
)
