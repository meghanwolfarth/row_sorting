id="z8ehh0"
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================================
# CONFIG
# ============================================

INPUT_FILE = "Diane Miller TwoRow_20260512_091535.csv"
OUTPUT_FILE = "orchard_final_sorted.csv"

TIME_GAP_THRESHOLD = 3.0

# GPS jump indicating new aisle
LON_JUMP_THRESHOLD = 0.00008

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv(INPUT_FILE)

# ============================================
# PARSE DATETIME
# ============================================

df["DateTime"] = pd.to_datetime(
    df["DateTime"],
    format="%Y-%m-%d_%H:%M:%S"
)

# ============================================
# SPLIT GPS
# ============================================

coords = df["Position"].str.split(",", expand=True)

df["Latitude"] = coords[0].astype(float)
df["Longitude"] = coords[1].astype(float)

# ============================================
# SORT BY TIME
# ============================================

df = df.sort_values("DateTime").reset_index(drop=True)

# ============================================
# DETECT NEW PASSES
# ============================================

df["TimeDiff"] = (
    df["DateTime"].diff().dt.total_seconds()
)

df["LonDiff"] = (
    df["Longitude"].diff().abs()
)

pass_ids = []
current_pass = 1

for i in range(len(df)):

    if i == 0:
        pass_ids.append(current_pass)
        continue

    new_pass = False

    if df.loc[i, "TimeDiff"] > TIME_GAP_THRESHOLD:
        new_pass = True

    elif df.loc[i, "LonDiff"] > LON_JUMP_THRESHOLD:
        new_pass = True

    if new_pass:
        current_pass += 1

    pass_ids.append(current_pass)

df["PassID"] = pass_ids

# ============================================
# SPLIT EACH PASS INTO 2 ROWS
# ============================================

all_rows = []

global_row_counter = 1

for pass_id in sorted(df["PassID"].unique()):

    subset = df[df["PassID"] == pass_id].copy()

    # Skip tiny accidental groups
    if len(subset) < 5:
        continue

    # ----------------------------------------
    # Cluster longitude into TWO rows
    # ----------------------------------------

    X = subset[["Longitude"]]

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    subset["SideCluster"] = kmeans.fit_predict(X)

    # Order west → east
    centers = (
        subset.groupby("SideCluster")["Longitude"]
        .mean()
        .sort_values()
    )

    cluster_map = {
        centers.index[0]: global_row_counter,
        centers.index[1]: global_row_counter + 1
    }

    subset["Row"] = subset["SideCluster"].map(cluster_map)

    # ----------------------------------------
    # Sort each row north/south
    # ----------------------------------------

    for row_num in sorted(subset["Row"].unique()):

        row_df = subset[
            subset["Row"] == row_num
        ].copy()

        # Determine travel direction
        lat_change = (
            row_df["Latitude"].iloc[-1]
            - row_df["Latitude"].iloc[0]
        )

        if lat_change > 0:
            row_df = row_df.sort_values("Latitude")
            direction = "SouthToNorth"
        else:
            row_df = row_df.sort_values(
                "Latitude",
                ascending=False
            )
            direction = "NorthToSouth"

        row_df["Direction"] = direction

        row_df["TreeNumber"] = range(
            1,
            len(row_df) + 1
        )

        all_rows.append(row_df)

    global_row_counter += 2

# ============================================
# COMBINE
# ============================================

final_df = pd.concat(all_rows)

# ============================================
# ASSIGN PLOTS
# ============================================

median_lat = final_df["Latitude"].median()

final_df["Plot"] = np.where(
    final_df["Latitude"] > median_lat,
    1,
    2
)

# ============================================
# CLEAN UP
# ============================================

final_df = final_df.drop(columns=[
    "Latitude",
    "Longitude",
    "TimeDiff",
    "LonDiff",
    "SideCluster"
])

# ============================================
# SORT OUTPUT
# ============================================

final_df = final_df.sort_values([
    "Row",
    "TreeNumber"
])

# ============================================
# SAVE
# ============================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Finished.")
print(f"Saved to {OUTPUT_FILE}")
