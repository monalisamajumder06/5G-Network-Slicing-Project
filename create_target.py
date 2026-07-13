import pandas as pd

# Load dataset
df = pd.read_csv("embb_dataset_v4.csv")

# Create target column
df["future_throughput_mbps"] = df["throughput_mbps"].shift(-1)

# Remove last row (it has no future value)
df = df.dropna()

# Save new dataset
df.to_csv("embb_dataset_v4_with_target.csv", index=False)

print(df.head())
print("\nShape:", df.shape)
