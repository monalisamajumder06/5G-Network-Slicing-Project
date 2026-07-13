import csv
import time

INTERFACES = [
    "ens33"
]

OUTPUT_FILE = "embb_dataset.csv"


def read_stats():
    stats = {}

    with open("/proc/net/dev", "r") as f:
        for line in f:
            if ":" not in line:
                continue

            iface, data = line.split(":", 1)
            iface = iface.strip()

            if iface not in INTERFACES:
                continue

            values = data.split()

            rx_bytes = int(values[0])
            tx_bytes = int(values[8])

            stats[iface] = (rx_bytes, tx_bytes)

    return stats


print("Starting eMBB metrics collector...")
print("Press Ctrl+C to stop")

previous = read_stats()

with open(OUTPUT_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "timestamp",
        "throughput_mbps",
        "active_users"
    ])

    try:

        while True:

            time.sleep(1)

            current = read_stats()

            total_bytes = 0

            for iface in INTERFACES:

                if iface not in previous:
                    continue

                if iface not in current:
                    continue

                prev_rx, prev_tx = previous[iface]
                curr_rx, curr_tx = current[iface]

                total_bytes += (curr_rx - prev_rx)
                total_bytes += (curr_tx - prev_tx)

            throughput_mbps = (total_bytes * 8) / 1_000_000

            active_users = len(current)

            writer.writerow([
                int(time.time()),
                round(throughput_mbps, 3),
                active_users
            ])

            file.flush()

            print(
    f"Bytes Delta: {total_bytes} | "
    f"Throughput: {throughput_mbps:.2f} Mbps | "
    f"Active Users: {active_users}"
)            

            previous = current

    except KeyboardInterrupt:
        print("\nCollection stopped.")
