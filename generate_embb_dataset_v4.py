import pandas as pd
import numpy as np

np.random.seed(42)

num_samples = 20000

timestamps = []
throughputs = []
active_users_list = []
packet_losses = []

# Initial values
active_users = 4
prev_packet_loss = 0
network_state = 0


for t in range(num_samples):

    
    # Active users random walk
    
    active_users += np.random.choice([-1, 0, 1])

    # Occasional burst
    if np.random.rand() < 0.01:
        active_users += np.random.randint(3, 7)

    active_users = np.clip(active_users, 1, 10)

    # Store history
    user_history.append(active_users)
    if len(user_history) > 10:
        user_history.pop(0)

   
    delayed_users = user_history[0]

    
    # Hidden network state
    
    network_state = (
        0.95 * network_state   #keep 95% of the previous network state
        + np.random.normal(0, 0.3) # add random fluctuations
    )

    
    # Persistent packet loss
    
    packet_loss = (
        0.8 * prev_packet_loss
        + 0.1 * active_users
        + np.random.normal(0, 0.2)
    )

    packet_loss = max(0, packet_loss)
    prev_packet_loss = packet_loss

   
    # Long-range periodic patterns
    # -----------------------------------
    pattern1 = 8 * np.sin(2 * np.pi * t / 200)
    pattern2 = 5 * np.sin(2 * np.pi * t / 500)

 
    # Throughput generation
    
    throughput = (
        100
        - 5 * delayed_users
        - 3 * packet_loss
        + 8 * network_state
        + pattern1
        + pattern2
        + np.random.normal(0, 1.5)
    )

    throughput = np.clip(throughput, 5, 100)

    timestamps.append(t)
    throughputs.append(throughput)
    active_users_list.append(active_users)
    packet_losses.append(packet_loss)

# Create DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "throughput_mbps": throughputs,
    "active_users": active_users_list,
    "packet_loss": packet_losses
})

# Target: next timestep throughput
df["future_throughput"] = df["throughput_mbps"].shift(-1)

# Remove last row
df.dropna(inplace=True)

# Save dataset
df.to_csv("embb_dataset_v4.csv", index=False)

print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns)
