import numpy as np
import pandas as pd
import os


TOTAL_PRBS = 100

MIN_PRBS = {
    "eMBB": 20,
    "URLLC": 30,
    "mMTC": 20
}


EMBB_FILE = os.path.expanduser(
    "~/embb_udp/embb_lstm_predictions.csv"
)

URLLC_FILE = os.path.expanduser(
    "~/urllc_latency/urllc_patchtst_predictions.csv"
)

MMTC_FILE = os.path.expanduser(
    "~/mmtc_udp/mmtc_patchtst_predictions.csv"
)


def calculate_demand(
    embb_throughput,
    urllc_latency,
    mmtc_packet_rate
):

    embb_demand = np.clip(
        embb_throughput / 10.0,
        0.0,
        1.0
    )

    urllc_demand = np.clip(
        urllc_latency / 3.0,
        0.0,
        1.0
    )

    mmtc_demand = np.clip(
        mmtc_packet_rate / 700.0,
        0.0,
        1.0
    )

    return {
        "eMBB": embb_demand,
        "URLLC": urllc_demand,
        "mMTC": mmtc_demand
    }


def allocate_prbs(demand):

    allocation = MIN_PRBS.copy()

    remaining = TOTAL_PRBS - sum(MIN_PRBS.values())

    total_demand = sum(demand.values())

    if total_demand > 0:

        for slice_name in demand:

            extra = (
                demand[slice_name] /
                total_demand
            ) * remaining

            allocation[slice_name] += extra

    allocation = {
        k: int(round(v))
        for k, v in allocation.items()
    }

    difference = (
        TOTAL_PRBS -
        sum(allocation.values())
    )

    allocation["URLLC"] += difference

    return allocation


# --------------------------------------------------
# LOAD PREDICTIONS
# --------------------------------------------------

embb_df = pd.read_csv(EMBB_FILE)
urllc_df = pd.read_csv(URLLC_FILE)
mmtc_df = pd.read_csv(MMTC_FILE)


print("\n==============================")
print("EXPERIMENT-AWARE RESOURCE CONTROLLER")
print("==============================")


# --------------------------------------------------
# COMMON EXPERIMENTS
# --------------------------------------------------

experiments = [
    "1UE",
    "2UE",
    "4UE",
    "6UE",
    "8UE",
    "10UE"
]


allocations = []


# --------------------------------------------------
# PROCESS EACH EXPERIMENT SEPARATELY
# --------------------------------------------------

for experiment in experiments:

    embb_exp = embb_df[
        embb_df["experiment"] == experiment
    ].reset_index(drop=True)

    urllc_exp = urllc_df[
        urllc_df["experiment"] == experiment
    ].reset_index(drop=True)

    mmtc_exp = mmtc_df[
        mmtc_df["experiment"] == experiment
    ].reset_index(drop=True)


    # Number of samples available in all three slices
    common_steps = min(
        len(embb_exp),
        len(urllc_exp),
        len(mmtc_exp)
    )


    print(
        f"\n{experiment}: "
        f"eMBB={len(embb_exp)}, "
        f"URLLC={len(urllc_exp)}, "
        f"mMTC={len(mmtc_exp)} "
        f"→ using {common_steps}"
    )


    # --------------------------------------------------
    # REPLAY COMMON PORTION
    # --------------------------------------------------

    for i in range(common_steps):

        embb_prediction = (
            embb_exp["Predicted"].iloc[i]
        )

        urllc_prediction = (
            urllc_exp["Predicted"].iloc[i]
        )

        mmtc_prediction = (
            mmtc_exp["Predicted"].iloc[i]
        )


        active_users = (
            embb_exp["active_users"].iloc[i]
        )


        demand = calculate_demand(
            embb_prediction,
            urllc_prediction,
            mmtc_prediction
        )


        allocation = allocate_prbs(demand)


        allocations.append({

            "experiment": experiment,

            "active_users": active_users,

            "step": i,

            "eMBB_prediction":
                embb_prediction,

            "URLLC_prediction":
                urllc_prediction,

            "mMTC_prediction":
                mmtc_prediction,

            "eMBB_demand":
                demand["eMBB"],

            "URLLC_demand":
                demand["URLLC"],

            "mMTC_demand":
                demand["mMTC"],

            "eMBB_PRBs":
                allocation["eMBB"],

            "URLLC_PRBs":
                allocation["URLLC"],

            "mMTC_PRBs":
                allocation["mMTC"]
        })


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

results = pd.DataFrame(allocations)


results.to_csv(
    "resource_allocation_results.csv",
    index=False
)


print("\n==============================")
print("RESULT")
print("==============================")

print(
    "Total controller decisions:",
    len(results)
)

print("\nDecisions per experiment:")

print(
    results.groupby("experiment").size()
)


print("\nFirst 10 decisions:")

print(
    results[
        [
            "experiment",
            "active_users",
            "step",
            "eMBB_prediction",
            "URLLC_prediction",
            "mMTC_prediction",
            "eMBB_PRBs",
            "URLLC_PRBs",
            "mMTC_PRBs"
        ]
    ].head(10).to_string(index=False)
)


print("\nSaved:")
print("resource_allocation_results.csv")

print("\n==============================")
print("CONTROLLER COMPLETE")
print("==============================")
