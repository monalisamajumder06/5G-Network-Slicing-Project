import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 20000

# Timestamp

timestamp = np.arange(n_samples)


# Active Users (Random Walk: 1-10)

active_users = [np.random.randint(1, 11)]

for _ in range(1, n_samples):
    change = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
    new_user = np.clip(active_users[-1] + change, 1, 10)
    active_users.append(new_user)

active_users = np.array(active_users)

#Congestion State

congestion = np.zeros(n_samples)

i = 0
while i < n_samples:

    if np.random.rand() < 0.01:      # 1% chance of congestion
        duration = np.random.randint(5, 20)

        end = min(i + duration, n_samples)
        congestion[i:end] = 1

        i = end
    else:
        i += 1


# Latency Generation

base_latency = 1.5 + 0.4 * active_users

latency = (
    base_latency
    + congestion * np.random.uniform(10, 30, n_samples)
    + np.random.normal(0, 0.5, n_samples)
)

latency = np.clip(latency, 1, None)


# Jitter Generation

jitter = (
    0.1 * latency
    + np.random.normal(0, 0.2, n_samples)
)

jitter = np.clip(jitter, 0, 8)


# Packet Loss (%)

packet_loss = (
    0.05 * latency
    + np.random.normal(0, 0.2, n_samples)
)

packet_loss = np.clip(packet_loss, 0, 5)

# Create DataFrame

df = pd.DataFrame({
    "timestamp": timestamp,
    "latency_ms": latency,
    "jitter_ms": jitter,
    "packet_loss": packet_loss,
    "active_users": active_users
})


# Target
df["future_latency_ms"] = df["latency_ms"].shift(-1)

# Remove last row
df.dropna(inplace=True)

# Save
df.to_csv("urllc_dataset.csv", index=False)

print(df.head())
print(df.describe())
print("\nDataset shape:", df.shape)
