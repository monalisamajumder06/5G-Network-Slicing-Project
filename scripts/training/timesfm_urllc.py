import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import timesfm



df = pd.read_csv("urllc_dataset_with_target.csv")

series = df["latency_ms"].values.astype(np.float32)


split_idx = int(len(series) * 0.8)

train_series = series[:split_idx]
test_series = series[split_idx:]


model = timesfm.TimesFM_2p5_200M_torch()

config = timesfm.ForecastConfig(
    max_context=128,
    max_horizon=1
)

model.compile(config)

print("TimesFM loaded successfully!")



predictions = []

for i in range(100):
    history = series[:split_idx + i]

    forecast, _ = model.forecast(
        horizon=1,
        inputs=[history.astype(np.float32)]
    )

    predictions.append(forecast[0][0])



actual = test_series[:100]

mae = mean_absolute_error(actual, predictions)

rmse = np.sqrt(
    mean_squared_error(actual, predictions)
)
print("\nResults:")
print("MAE :", mae)
print("RMSE:", rmse)
