import pandas as pd
import numpy as np

# Number of samples
num_samples = 20000

# Lists to store data
timestamps = []
packet_rates = []
active_users_list = []
packet_losses = []

# Initial number of active users
active_users = np.random.randint(20, 60)

# Burst state variables
burst_remaining = 0
burst_value = 0
burst_duration = 0

for t in range(num_samples):

    
    # Active users evolution
   
    change = np.random.choice(
        [-10, -1, 0, 1, 10],
        p=[0.02, 0.24, 0.48, 0.24, 0.02]
    )

    active_users += change
    active_users = np.clip(active_users, 1, 100)

    # -----------------------------
    # Periodic traffic component
    # -----------------------------
    periodic = 100 * np.sin(2 * np.pi * t / 500)

    # -----------------------------
    # Congestion indicator
    # -----------------------------
    congestion = active_users / 100

    # -----------------------------
    # Burst probability depends on congestion
    # -----------------------------
    burst_probability = 0.01

    if active_users > 70:
        burst_probability = 0.05

    if active_users > 85:
        burst_probability = 0.10

    # -----------------------------
    # Start new burst
    # -----------------------------
    if (active_users > 85 and packet_loss > 2.2 and burst_remaining == 0):

        burst_duration = np.random.randint(5, 12)

        burst_remaining = burst_duration

        burst_value = np.random.uniform(150, 450)

    # -----------------------------
    # Generate burst (ramp up + ramp down)
    # -----------------------------
    if burst_remaining > 0:

        progress = (
            burst_duration - burst_remaining
        ) / burst_duration

        if progress < 0.5:
            burst = burst_value * (progress * 2)
        else:
            burst = burst_value * (2 - progress * 2)

        burst_remaining -= 1

    else:
        burst = 0

    # -----------------------------
    # Random noise
    # -----------------------------
    noise = np.random.normal(0, 20)

    # -----------------------------
    # Packet rate generation
    # -----------------------------
    packet_rate = (
        8 * active_users
        + periodic
        + burst
        + noise
    )

    packet_rate = max(10, packet_rate)

    # -----------------------------
    # Packet loss generation
    # -----------------------------
    packet_loss = (
        0.5
        + 0.02 * active_users
        + 0.5 * congestion
        + np.random.normal(0, 0.3)
    )

    packet_loss = np.clip(packet_loss, 0, 5)

    # -----------------------------
    # Store values
    # -----------------------------
    timestamps.append(t)
    packet_rates.append(packet_rate)
    active_users_list.append(active_users)
    packet_losses.append(packet_loss)

# -----------------------------
# Create DataFrame
# -----------------------------
df = pd.DataFrame({
    "timestamp": timestamps,
    "packet_rate": packet_rates,
    "active_users": active_users_list,
    "packet_loss": packet_losses
})

# -----------------------------
# Create target column
# -----------------------------
df["future_packet_rate"] = df["packet_rate"].shift(-1)

# Remove last row
df.dropna(inplace=True)

# -----------------------------
# Save dataset
# -----------------------------
df.to_csv("mmtc_dataset_v2.csv", index=False)

print(df.head())
print("\nDataset shape:", df.shape)
print("\nStatistics:")
print(df.describe())
