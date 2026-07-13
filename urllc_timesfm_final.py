import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timesfm

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print("Loading dataset...")

df = pd.read_csv("urllc_dataset_with_target.csv")

latency = df["latency_ms"].values
jitter = df["jitter_ms"].values
packet_loss = df["packet_loss"].values
active_users = df["active_users"].values

split_idx = int(len(latency) * 0.8)

print("Loading TimesFM...")

model = timesfm.TimesFM_2p5_200M_torch()

config = timesfm.ForecastConfig(
    max_context=128,
    max_horizon=1,
    return_backcast=True
)

model.compile(config)

print("Starting evaluation...")

predictions = []
actuals = []

end_idx = min(split_idx + 4000, len(latency))

for i in range(split_idx, end_idx):

    history = latency[i-128:i]

    covariates = {
        "jitter": [jitter[i-128:i+1].tolist()],
        "packet_loss": [packet_loss[i-128:i+1].tolist()],
        "active_users": [active_users[i-128:i+1].tolist()]
    }

    result = model.forecast_with_covariates(
        inputs=[history.tolist()],
        dynamic_numerical_covariates=covariates,
        xreg_mode="xreg + timesfm"
    )

    prediction = result[0][0][0]

    predictions.append(prediction)
    actuals.append(latency[i])

    if (i - split_idx + 1) % 10 == 0:
        print(f"Completed {i - split_idx + 1}/{end_idx - split_idx}")

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

plt.title("TimesFM: Actual vs Predicted Latency")
plt.xlabel("Test Samples")
plt.ylabel("Latency (ms)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("timesfm_actual_vs_predicted_urllc.png", dpi=300)
plt.show()
