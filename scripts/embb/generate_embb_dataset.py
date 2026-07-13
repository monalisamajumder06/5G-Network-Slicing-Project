import numpy as np
import pandas as pd

np.random.seed(42)

N = 5000

t = np.arange(N)

# Daily-like traffic pattern
base = 60 + 35 * np.sin(2 * np.pi * t / 1000)

# Bursts
bursts = np.zeros(N)

for _ in range(25):
    start = np.random.randint(0, N - 100)
    duration = np.random.randint(20, 100)
    intensity = np.random.uniform(15, 50)

    bursts[start:start + duration] += intensity

# Small fluctuations
noise = np.random.normal(0, 3, N)

throughput = np.clip(
    base + bursts + noise,
    20,
    160
)

active_users = []

for tp in throughput:

    if tp < 40:
        active_users.append(1)

    elif tp < 80:
        active_users.append(2)

    elif tp < 120:
        active_users.append(3)

    else:
        active_users.append(4)

df = pd.DataFrame({
    "timestamp": np.arange(N),
    "throughput_mbps": np.round(throughput, 2),
    "active_users": active_users
})

df.to_csv("embb_dataset.csv", index=False)

print("eMBB dataset generated successfully.")
print(df.head())
