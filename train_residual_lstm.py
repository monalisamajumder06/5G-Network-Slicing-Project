import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Load residual dataset
# -----------------------------
df = pd.read_csv("timesfm_residual_dataset.csv")

# Features
X = df[
    [
        "timesfm_prediction",
        "active_users",
        "packet_loss"
    ]
].values

# Target
y = df["residual"].values.reshape(-1, 1)

# -----------------------------
# Train/Test split
# -----------------------------
split_idx = int(len(df) * 0.8)

X_train = X[:split_idx]
X_test = X[split_idx:]

y_train = y[:split_idx]
y_test = y[split_idx:]

# -----------------------------
# Scaling
# -----------------------------
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

joblib.dump(scaler_x, "residual_scaler_x.pkl")
joblib.dump(scaler_y, "residual_scaler_y.pkl")

# -----------------------------
# Sequence creation
# -----------------------------
sequence_length = 10

def create_sequences(X, y):

    xs = []
    ys = []

    for i in range(len(X) - sequence_length):

        xs.append(X[i:i+sequence_length])
        ys.append(y[i+sequence_length])

    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(X_train, y_train)
X_test, y_test = create_sequences(X_test, y_test)

# -----------------------------
# Tensor conversion
# -----------------------------
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)

# -----------------------------
# LSTM Model
# -----------------------------
class ResidualLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(64,1)

    def forward(self,x):

        out,_ = self.lstm(x)

        out = out[:,-1,:]

        return self.fc(out)

model = ResidualLSTM()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------------------
# Training
# -----------------------------
epochs = 100

for epoch in range(epochs):

    model.train()

    for X_batch,y_batch in train_loader:

        optimizer.zero_grad()

        pred = model(X_batch)

        loss = criterion(pred,y_batch)

        loss.backward()

        optimizer.step()

    if (epoch+1)%10==0:

        print(f"Epoch {epoch+1}/{epochs} Loss {loss.item():.6f}")

# -----------------------------
# Save model
# -----------------------------
torch.save(
    model.state_dict(),
    "timesfm_lstm_residual.pth"
)

# -----------------------------
# Evaluation
# -----------------------------
model.eval()

with torch.no_grad():

    pred = model(X_test).numpy()

pred = scaler_y.inverse_transform(pred)
y_test = scaler_y.inverse_transform(y_test.numpy())

mae = mean_absolute_error(y_test,pred)
rmse = np.sqrt(mean_squared_error(y_test,pred))

print("\nResidual LSTM")
print("MAE :",mae)
print("RMSE:",rmse)
