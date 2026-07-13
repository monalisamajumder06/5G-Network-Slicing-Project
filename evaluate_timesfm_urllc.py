import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("urllc_timesfm_predictions.csv")

plt.figure(figsize=(12,5))

plt.plot(df["Actual"][:200], label="Actual")
plt.plot(df["Predicted"][:200], label="Predicted")

plt.title("TimesFM Actual vs Predicted")
plt.xlabel("Sample")
plt.ylabel("Latency (ms)")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("timesfm_actual_vs_predicted.png", dpi=300)
plt.show()
