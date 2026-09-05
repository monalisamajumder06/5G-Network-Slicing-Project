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
import random
import numpy as np


seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load dataset
df = pd.read_csv("urllc_dataset_with_target.csv")

# Features and target
features = df[["latency_ms", "jitter_ms","packet_loss", "active_users"]].values
target = df["future_latency_ms"].values

# Sliding window
sequence_length = 5

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
X_train_reshaped = X_train.reshape(-1, 4)
X_test_reshaped = X_test.reshape(-1, 4)

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
    batch_size=64,
    shuffle=False
)
#positional encoding
import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.arange(
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        pe = pe.repeat(
            1,
            d_model
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        x = x + self.pe[:, :x.size(1), :]

        return x
# Transformer Model
import torch
import torch.nn as nn

class PatchTST(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_size = 5
        self.stride = 5

        
        self.input_projection = nn.Linear(
            20,
            128
        )

        self.pos_encoder = PositionalEncoding(128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dropout=0.05,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(128, 1)

    def forward(self, x):

        batch_size = x.shape[0]
        x = x.unfold(
            dimension=1,
            size=self.patch_size,
            step=self.stride
        )

        

        x = x.permute(0, 1, 3, 2)

        

        x = x.reshape(x.size(0), -1, 20)
        

        x = self.input_projection(x)

        x = self.pos_encoder(x)
        
        x = self.transformer(x)

        # Use last patch representation
        x = x[:, -1, :]

        x = self.fc(x)

        return x
model = PatchTST()
print("Training shape:", X_train_tensor.shape)

sample_output = model(X_train_tensor[:32])

print("Output shape:", sample_output.shape)
# Loss Function
criterion = nn.L1Loss()

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay = 1e-4
)

# Training
epochs = 75

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
plt.ylabel("Latency (ms)")

plt.title(
    "Actual vs Predicted Latency"
)

plt.legend()

plt.grid(True)

plt.show()

# Save Model
torch.save(
    model.state_dict(),
    "urllc_transformer.pth"
)

print(
    "\nModel saved as urllc_transformer.pth"
)
