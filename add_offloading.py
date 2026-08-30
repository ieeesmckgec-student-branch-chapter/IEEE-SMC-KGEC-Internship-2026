import random
import numpy as np
import pandas as pd

from simulation.task_generator import generate_tasks

from optimizers.baseline import baseline
from optimizers.pso import pso
from optimizers.de import de
from optimizers.ga import ga
from optimizers.lion import lion
from optimizers.hybrid_ga_lion import hybrid_ga_lion


# ==========================================================
# CONFIGURATION
# ==========================================================

RUNS = 20

INPUT_FILE = "results.csv"
OUTPUT_FILE = "main.csv"


ALGORITHMS = {
    "Baseline": baseline,
    "PSO": pso,
    "DE": de,
    "GA": ga,
    "Lion": lion,
    "Hybrid": hybrid_ga_lion
}


# ==========================================================
# WORKLOAD RANGE
# ==========================================================

def get_workload_range(workload):

    if workload <= 15000:

        return "5k-15k"

    elif workload <= 35000:

        return "20k-35k"

    else:

        return "40k-50k"


# ==========================================================
# CALCULATE OFFLOADING PERCENTAGE
# ==========================================================

def calculate_offloading(allocation):

    total_tasks = len(allocation)

    if total_tasks == 0:

        return 0.0, 0.0

    cloud_tasks = sum(
        1
        for server in allocation
        if server in [
            "CLOUD1",
            "CLOUD2"
        ]
    )

    edge_tasks = sum(
        1
        for server in allocation
        if server in [
            "EDGE1",
            "EDGE2",
            "EDGE3"
        ]
    )

    cloud_percentage = (
        cloud_tasks / total_tasks
    ) * 100

    edge_percentage = (
        edge_tasks / total_tasks
    ) * 100

    return (
        cloud_percentage,
        edge_percentage
    )


# ==========================================================
# LOAD EXISTING RESULTS
# ==========================================================

print("=" * 80)

print(
    "CREATING FINAL MENTOR-FACING MAIN.CSV"
)

print("=" * 80)


try:

    results_df = pd.read_csv(
        INPUT_FILE
    )

except FileNotFoundError:

    print(
        f"\nERROR: {INPUT_FILE} was not found."
    )

    print(
        "\nMake sure results.csv is in the "
        "same folder as this script."
    )

    raise SystemExit


print(
    f"\nLoaded: {INPUT_FILE}"
)

print(
    f"Rows found: {len(results_df)}"
)


# ==========================================================
# CHECK REQUIRED COLUMN
# ==========================================================

if "Tasks" not in results_df.columns:

    print(
        "\nERROR: 'Tasks' column was not found "
        "in results.csv."
    )

    raise SystemExit


# ==========================================================
# GET WORKLOADS DIRECTLY FROM RESULTS.CSV
# ==========================================================

WORKLOADS = (
    results_df["Tasks"]
    .astype(int)
    .tolist()
)


print(
    "\nWorkloads found:"
)

print(
    WORKLOADS
)


# ==========================================================
# OFFLOADING RESULTS
# ==========================================================

offloading_results = []


# ==========================================================
# REPRODUCE OFFLOADING EXPERIMENT
# ==========================================================

for workload in WORKLOADS:

    workload_range = (
        get_workload_range(workload)
    )

    print(
        "\n" +
        "=" * 70
    )

    print(
        f"Processing workload: "
        f"{workload} "
        f"({workload_range})"
    )

    print(
        "=" * 70
    )


    # ------------------------------------------------------
    # Store 20 runs for each algorithm
    # ------------------------------------------------------

    algorithm_results = {}

    for algorithm_name in ALGORITHMS:

        algorithm_results[
            algorithm_name
        ] = {

            "cloud": [],

            "edge": []
        }


    # ======================================================
    # RUN 20 EXPERIMENTS
    # ======================================================

    for run in range(RUNS):

        # --------------------------------------------------
        # Reproducible seed
        # --------------------------------------------------

        seed = (
            42
            + workload
            + run
        )

        random.seed(seed)

        np.random.seed(seed)


        # --------------------------------------------------
        # Generate task set
        # --------------------------------------------------

        tasks = generate_tasks(
            max(
                50,
                workload // 100
            )
        )


        print(
            f"Run {run + 1:02d}/{RUNS}",
            end=" | "
        )


        # ==================================================
        # RUN EVERY ALGORITHM
        # ==================================================

        for (
            algorithm_name,
            algorithm
        ) in ALGORITHMS.items():

            # --------------------------------------------------
            # Run algorithm
            # --------------------------------------------------

            allocation = algorithm(
                tasks
            )


            # --------------------------------------------------
            # Calculate Edge / Cloud percentages
            # --------------------------------------------------

            cloud_percentage, edge_percentage = (
                calculate_offloading(
                    allocation
                )
            )


            # --------------------------------------------------
            # Store result
            # --------------------------------------------------

            algorithm_results[
                algorithm_name
            ][
                "cloud"
            ].append(
                cloud_percentage
            )


            algorithm_results[
                algorithm_name
            ][
                "edge"
            ].append(
                edge_percentage
            )


        print(
            "completed"
        )


    # ======================================================
    # AVERAGE 20 RUNS
    # ======================================================

    for algorithm_name in ALGORITHMS:

        average_cloud = np.mean(
            algorithm_results[
                algorithm_name
            ][
                "cloud"
            ]
        )

        average_edge = np.mean(
            algorithm_results[
                algorithm_name
            ][
                "edge"
            ]
        )


        offloading_results.append({

            "Workload": workload,

            "Workload Range": workload_range,

            "Algorithm": algorithm_name,

            "Cloud Offloading %":
                average_cloud,

            "Edge Processing %":
                average_edge
        })


# ==========================================================
# CREATE OFFLOADING DATAFRAME
# ==========================================================

offloading_df = pd.DataFrame(
    offloading_results
)


# ==========================================================
# ROUND OFFLOADING VALUES
# ==========================================================

offloading_df[
    "Cloud Offloading %"
] = offloading_df[
    "Cloud Offloading %"
].round(4)


offloading_df[
    "Edge Processing %"
] = offloading_df[
    "Edge Processing %"
].round(4)


# ==========================================================
# CREATE FINAL MAIN.CSV
# ==========================================================

final_rows = []


# ==========================================================
# CONVERT WIDE RESULTS INTO MENTOR-FRIENDLY FORMAT
# ==========================================================

for _, result in results_df.iterrows():

    workload = int(
        result["Tasks"]
    )

    workload_range = (
        get_workload_range(
            workload
        )
    )


    for algorithm_name in ALGORITHMS:

        # --------------------------------------------------
        # Find offloading result
        # --------------------------------------------------

        matching_offloading = (
            offloading_df[
                (
                    offloading_df[
                        "Workload"
                    ]
                    == workload
                )
                &
                (
                    offloading_df[
                        "Algorithm"
                    ]
                    == algorithm_name
                )
            ]
        )


        if matching_offloading.empty:

            print(
                "\nWARNING: Offloading result "
                "not found for:",
                workload,
                algorithm_name
            )

            continue


        offloading_row = (
            matching_offloading.iloc[0]
        )


        # --------------------------------------------------
        # Add final row
        # --------------------------------------------------

        final_rows.append({

            "Workload Range":
                workload_range,

            "Workload":
                workload,

            "Algorithm":
                algorithm_name,

            "Response Time":
                result[
                    f"{algorithm_name} Response"
                ],

            "Energy Consumption":
                result[
                    f"{algorithm_name} Energy"
                ],

            "Cloud Offloading %":
                offloading_row[
                    "Cloud Offloading %"
                ],

            "Edge Processing %":
                offloading_row[
                    "Edge Processing %"
                ],

            "Latency":
                result[
                    f"{algorithm_name} Latency"
                ],

            "Throughput":
                result[
                    f"{algorithm_name} Throughput"
                ],

            "Cost":
                result[
                    f"{algorithm_name} Cost"
                ],

            "SLA Violations":
                result[
                    f"{algorithm_name} SLA"
                ],

            "Runtime (s)":
                result[
                    f"{algorithm_name} Runtime"
                ]
        })


# ==========================================================
# CREATE FINAL DATAFRAME
# ==========================================================

main_df = pd.DataFrame(
    final_rows
)


# ==========================================================
# REMOVE FITNESS IF PRESENT
# ==========================================================

fitness_columns = [

    column

    for column in main_df.columns

    if "fitness" in column.lower()
]


if fitness_columns:

    main_df.drop(
        columns=fitness_columns,
        inplace=True
    )


# ==========================================================
# ROUND NUMERIC VALUES
# ==========================================================

numeric_columns = [

    "Response Time",

    "Energy Consumption",

    "Cloud Offloading %",

    "Edge Processing %",

    "Latency",

    "Throughput",

    "Cost",

    "SLA Violations",

    "Runtime (s)"
]


for column in numeric_columns:

    if column in main_df.columns:

        main_df[column] = pd.to_numeric(
            main_df[column],
            errors="coerce"
        ).round(4)


# ==========================================================
# VALIDATE OFFLOADING
# ==========================================================

print(
    "\n" +
    "=" * 80
)

print(
    "OFFLOADING VALIDATION"
)

print(
    "=" * 80
)


main_df[
    "Offloading Total %"
] = (

    main_df[
        "Cloud Offloading %"
    ]

    +

    main_df[
        "Edge Processing %"
    ]
)


maximum_difference = (

    main_df[
        "Offloading Total %"
    ]

    - 100

).abs().max()


print(
    "\nMaximum difference from 100%:",
    f"{maximum_difference:.6f}%"
)


if maximum_difference < 0.01:

    print(
        "\nPASS:"
        " Cloud Offloading % +"
        " Edge Processing % = 100%"
    )

else:

    print(
        "\nWARNING:"
        " Some rows do not sum to 100%."
    )


# ----------------------------------------------------------
# Remove temporary validation column
# ----------------------------------------------------------

main_df.drop(
    columns=[
        "Offloading Total %"
    ],
    inplace=True
)


# ==========================================================
# FINAL COLUMN ORDER
# ==========================================================

final_column_order = [

    "Workload Range",

    "Workload",

    "Algorithm",

    "Response Time",

    "Energy Consumption",

    "Cloud Offloading %",

    "Edge Processing %",

    "Latency",

    "Throughput",

    "Cost",

    "SLA Violations",

    "Runtime (s)"
]


main_df = main_df[
    final_column_order
]


# ==========================================================
# SAVE MAIN.CSV
# ==========================================================

main_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================================
# FINAL INFORMATION
# ==========================================================

print(
    "\n" +
    "=" * 80
)

print(
    "MAIN.CSV GENERATED SUCCESSFULLY"
)

print(
    "=" * 80
)


print(
    "\nFile:",
    OUTPUT_FILE
)

print(
    "Rows:",
    len(main_df)
)

print(
    "Columns:",
    len(main_df.columns)
)


print(
    "\nColumns included:"
)

for column in main_df.columns:

    print(
        " -",
        column
    )


# ==========================================================
# SHOW FIRST RESULTS
# ==========================================================

print(
    "\nFirst 12 rows:"
)

print(
    main_df.head(12).to_string(
        index=False
    )
)


# ==========================================================
# FINAL MESSAGE
# ==========================================================

print(
    "\n" +
    "=" * 80
)

print(
    "DONE"
)

print(
    "=" * 80
)