import pandas as pd
import numpy as np
import timesfm

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print("Loading dataset...")

df = pd.read_csv("embb_dataset_v4.csv")

throughput = df["throughput_mbps"].values

split_idx = int(len(df) * 0.8)

print("Loading TimesFM...")

model = timesfm.TimesFM_2p5_200M_torch()

config = timesfm.ForecastConfig(
    max_context=128,
    max_horizon=1,
    return_backcast=True,
    normalize_inputs=True
)

model.compile(config)

print("Starting evaluation...")

predictions = []
actuals = []

for i in range(max(split_idx, 128), len(df)):

    history = throughput[i-128:i]

    # Plain TimesFM (NO covariates)
    forecast = model.forecast(
        inputs=[history.tolist()],
        horizon=1
    )

    prediction = forecast[0][0][0]

    predictions.append(prediction)
    actuals.append(df["future_throughput"].iloc[i])

    if (i - split_idx + 1) % 100 == 0:
        print(f"Completed {i - split_idx + 1}/{len(df) - split_idx}")

mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))

print("\n========================")
print("TimesFM Results")
print("========================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# Persistence Baseline
baseline = throughput[split_idx-1:-1]

baseline_mae = mean_absolute_error(actuals, baseline)
baseline_rmse = np.sqrt(mean_squared_error(actuals, baseline))

print("\nPersistence Baseline")
print(f"MAE  : {baseline_mae:.4f}")
print(f"RMSE : {baseline_rmse:.4f}")
