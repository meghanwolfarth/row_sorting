import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =====================================================
# SETTINGS
# =====================================================

INPUT_FILE = "Crop_Data_Hort 2_20260416_100400.csv"
OUTPUT_FILE = "USDA_final_sorted.csv"

EXPECTED_ROWS = 5

# Tree filtering thresholds
MIN_WIDTH = 5.0
MIN_DENSITY = 0.02

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(INPUT_FILE)

# =====================================================
# PARSE COORDINATES
# =====================================================

coords = df["Position"].str.split(",", expand=True)

df["latitude"] = coords[0].astype(float)
df["longitude"] = coords[1].astype(float)

# =====================================================
# PARSE TIMESTAMP
# =====================================================

df["DateTime"] = pd.to_datetime(
    df["DateTime"],
    format="%Y-%m-%d_%H:%M:%S"
)

df = df.sort_values("DateTime").reset_index(drop=True)

# =====================================================
# REMOVE LIKELY NON-TREES
# =====================================================

trees = df[
    (df["Width (Feet)"] >= MIN_WIDTH) &
    (df["Density (Gallons)"] >= MIN_DENSITY)
].copy()

print(f"Original points: {len(df)}")
print(f"Tree points: {len(trees)}")

# =====================================================
# PCA ROTATION
# =====================================================

X = trees[["longitude", "latitude"]].values

pca = PCA(n_components=2)
rotated = pca.fit_transform(X)

trees["along_row"] = rotated[:, 0]
trees["cross_row"] = rotated[:, 1]

# =====================================================
# FIND ORCHARD ROWS
# =====================================================

kmeans = KMeans(
    n_clusters=EXPECTED_ROWS,
    random_state=42,
    n_init=25
)

trees["cluster"] = kmeans.fit_predict(
    trees[["cross_row"]]
)

# =====================================================
# ORDER ROWS SOUTH -> NORTH
# =====================================================

row_centers = (
    trees.groupby("cluster")["latitude"]
         .mean()
         .sort_values()
)

row_lookup = {
    cluster: row_num + 1
    for row_num, cluster
    in enumerate(row_centers.index)
}

trees["row"] = trees["cluster"].map(row_lookup)

# =====================================================
# NUMBER TREES WEST -> EAST
# =====================================================

final_rows = []

for row in sorted(trees["row"].unique()):

    row_df = trees[
        trees["row"] == row
    ].copy()

    row_df = row_df.sort_values(
        "along_row"
    )

    row_df["tree_number"] = np.arange(
        1,
        len(row_df) + 1
    )

    final_rows.append(row_df)

trees = pd.concat(final_rows)

# =====================================================
# ASSIGN ORCHARD GROUPS
# =====================================================

trees["group"] = np.where(
    trees["row"] <= 3,
    1,
    2
)

# =====================================================
# SUMMARY
# =====================================================

print("\nROW SUMMARY")
print("-" * 40)

summary = (
    trees.groupby("row")
         .size()
         .reset_index(name="tree_count")
)

print(summary)

print("\nExpected pattern:")
print("Rows 1-3 ≈ 30 trees")
print("Rows 4-5 ≈ 25 trees")

# =====================================================
# CLEAN OUTPUT
# =====================================================

output_cols = [
    "DateTime",
    "latitude",
    "longitude",
    "Height (Feet)",
    "Width (Feet)",
    "Density (Gallons)",
    "row",
    "group",
    "tree_number"
]

trees = trees[output_cols]

trees = trees.sort_values(
    ["row", "tree_number"]
)

trees.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved: {OUTPUT_FILE}")