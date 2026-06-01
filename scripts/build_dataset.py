"""
build_dataset.py
Enriches the existing twitch_structural.csv (4 structural features) with
5 pairwise node attribute features from musae_ENGB_target.csv.

Input:  twitch_structural.csv   (from, to, common_neighbors, jaccard_coefficient,
                                  adamic_adar, preferential_attachment, link)
        musae_ENGB_target.csv   (new_id, days, mature, views, partner)

Output: twitch_full.csv         (all 4 original + 5 new pairwise features)
"""

import pandas as pd
import numpy as np

# ── Load original dataset ──────────────────────────────────────────────────────
print("Loading twitch_structural.csv...")
df = pd.read_csv("twitch_structural.csv")
print(f"  Rows: {len(df):,}  Columns: {list(df.columns)}")

# ── Load node attributes ───────────────────────────────────────────────────────
print("Loading musae_ENGB_target.csv...")
df_target = pd.read_csv("musae_ENGB_target.csv").set_index("new_id")
print(f"  Node attribute rows: {len(df_target):,}")

# ── Compute pairwise node features ─────────────────────────────────────────────
print("Computing pairwise node features...")

def get_attr(node, col):
    return df_target.loc[node, col]

views_u   = df["from"].map(lambda n: df_target.loc[n, "views"])
views_v   = df["to"].map(lambda n:   df_target.loc[n, "views"])
days_u    = df["from"].map(lambda n: df_target.loc[n, "days"])
days_v    = df["to"].map(lambda n:   df_target.loc[n, "days"])
partner_u = df["from"].map(lambda n: int(str(df_target.loc[n, "partner"]) == "True"))
partner_v = df["to"].map(lambda n:   int(str(df_target.loc[n, "partner"]) == "True"))
mature_u  = df["from"].map(lambda n: int(str(df_target.loc[n, "mature"])  == "True"))
mature_v  = df["to"].map(lambda n:   int(str(df_target.loc[n, "mature"])  == "True"))

df["views_diff"]   = (views_u - views_v).abs()
df["views_ratio"]  = np.minimum(views_u, views_v) / (np.maximum(views_u, views_v) + 1e-9)
df["age_diff"]     = (days_u - days_v).abs()
df["same_partner"] = (partner_u == partner_v).astype(int)
df["same_mature"]  = (mature_u  == mature_v).astype(int)

# ── Save ───────────────────────────────────────────────────────────────────────
df.to_csv("twitch_full.csv", index=False)
print(f"\nSaved twitch_full.csv")
print(f"  Rows:    {len(df):,}")
print(f"  Columns: {list(df.columns)}")
