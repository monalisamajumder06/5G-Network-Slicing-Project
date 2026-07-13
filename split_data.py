import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# Load dataset
df = pd.read_csv("mmtc_dataset_v2_with_target.csv")

features = ["packet_rate", "active_users", "packet_loss"]
target = "future_packet_rate"

sequence_length = 10

X_data = df[features].values
y_data = df[target].values

sequence_length = 60

X = []
y = []

for i in range(len(X_data) - sequence_length):
    X.append(X_data[i:i + sequence_length])
    y.append(y_data[i + sequence_length])

X = np.array(X)
y = np.array(y)
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=False
)
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

print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)
print("y_test mean:", np.mean(y_test))
print("y_test std:", np.std(y_test))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
