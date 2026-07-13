import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timesfm

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print("Loading dataset...")

df = pd.read_csv("mmtc_dataset_v2_with_target.csv")

# -----------------------------
# Feature Engineering
# -----------------------------
df["sin_time"] = np.sin(2 * np.pi * df["timestamp"] / 500)
df["cos_time"] = np.cos(2 * np.pi * df["timestamp"] / 500)

df["burst_indicator"] = (
    (df["active_users"] > 85) &
    (df["packet_loss"] > 2.2)
).astype(int)

df["lag1"] = df["packet_rate"].shift(1)
df["lag2"] = df["packet_rate"].shift(2)
df["lag5"] = df["packet_rate"].shift(5)

df.dropna(inplace=True)

packet_rate = df["packet_rate"].values
packet_loss = df["packet_loss"].values
active_users = df["active_users"].values

sin_time = df["sin_time"].values
cos_time = df["cos_time"].values
burst_indicator = df["burst_indicator"].values
lag1 = df["lag1"].values
lag2 = df["lag2"].values
lag5 = df["lag5"].values

split_idx = int(len(packet_rate) * 0.8)

print("Loading TimesFM...")

model = timesfm.TimesFM_2p5_200M_torch()

config = timesfm.ForecastConfig(
    max_context=256,
    max_horizon=1,
    return_backcast=True
)

model.compile(config)

print("Starting evaluation...")

predictions = []
actuals = []

end_idx = min(split_idx + 4000, len(packet_rate))

for i in range(split_idx, end_idx):

    history = packet_rate[i-256:i]

    covariates = {
        "packet_loss": [packet_loss[i-256:i+1].tolist()],
        "active_users": [active_users[i-256:i+1].tolist()],
        "sin_time": [sin_time[i-256:i+1].tolist()],
        "cos_time": [cos_time[i-256:i+1].tolist()],
        "burst_indicator": [burst_indicator[i-256:i+1].tolist()],
        "lag1": [lag1[i-256:i+1].tolist()],
        "lag2": [lag2[i-256:i+1].tolist()],
        "lag5": [lag5[i-256:i+1].tolist()]
    }

    result = model.forecast_with_covariates(
        inputs=[history.tolist()],
        dynamic_numerical_covariates=covariates,
        xreg_mode="xreg + timesfm"
    )

    prediction = result[0][0][0]

    predictions.append(prediction)
    actuals.append(packet_rate[i])

    if (i - split_idx + 1) % 10 == 0:
        print(f"Completed {i - split_idx + 1}/4000")

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))

print("\nRESULTS")
print("MAE :", mae)
print("RMSE:", rmse)

# -----------------------------
# Actual vs Predicted Graph
# -----------------------------
plt.figure(figsize=(14,6))

plt.plot(actuals,
         label="Actual",
         linewidth=2)

plt.plot(predictions,
         label="Predicted",
         linewidth=2)

plt.title("TimesFM: Actual vs Predicted Packet Rate")
plt.xlabel("Test Samples")
plt.ylabel("Packet Rate")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("timesfm_actual_vs_predicted.png", dpi=300)
plt.show()
