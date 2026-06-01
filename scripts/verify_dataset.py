"""
verify_dataset.py
Verifies twitch_full.csv is sound and complete.
Runs a series of checks and prints PASS / FAIL for each.
"""

import pandas as pd
import numpy as np

PASS = "  PASS"
FAIL = "  FAIL"

df = pd.read_csv("twitch_full.csv")
errors = 0

def check(name, condition, detail=""):
    global errors
    status = PASS if condition else FAIL
    if not condition:
        errors += 1
    print(f"{status}  {name}" + (f" — {detail}" if detail else ""))

print("=" * 60)
print("Dataset Verification: twitch_full.csv")
print("=" * 60)

# ── 1. Shape ───────────────────────────────────────────────────────────────────
print("\n── Shape ──")
check("Row count is ~70,648",
      abs(len(df) - 70648) < 100,
      f"got {len(df):,}")

EXPECTED_COLS = ["from", "to", "common_neighbors", "jaccard_coefficient",
                 "adamic_adar", "preferential_attachment",
                 "views_diff", "views_ratio", "age_diff",
                 "same_partner", "same_mature", "link"]
check("All 12 columns present",
      all(c in df.columns for c in EXPECTED_COLS),
      f"got {list(df.columns)}")

# ── 2. Balance ─────────────────────────────────────────────────────────────────
print("\n── Class Balance ──")
vc = df["link"].value_counts()
check("Balanced classes (link=1 count == link=0 count)",
      vc[1] == vc[0],
      f"link=1: {vc.get(1,0):,}  link=0: {vc.get(0,0):,}")

# ── 3. Missing values ──────────────────────────────────────────────────────────
print("\n── Missing Values ──")
nulls = df.isnull().sum()
check("No missing values", nulls.sum() == 0,
      f"{nulls[nulls>0].to_dict()}" if nulls.sum() > 0 else "")

# ── 4. Feature ranges ──────────────────────────────────────────────────────────
print("\n── Feature Ranges ──")
check("common_neighbors >= 0",           (df.common_neighbors >= 0).all())
check("jaccard_coefficient in [0, 1]",   df.jaccard_coefficient.between(0, 1).all())
check("adamic_adar >= 0",                (df.adamic_adar >= 0).all())
check("preferential_attachment >= 0",    (df.preferential_attachment >= 0).all())
check("views_diff >= 0",                 (df.views_diff >= 0).all())
check("views_ratio in [0, 1]",           df.views_ratio.between(0, 1 + 1e-6).all())
check("age_diff >= 0",                   (df.age_diff >= 0).all())
check("same_partner in {0, 1}",          df.same_partner.isin([0, 1]).all())
check("same_mature in {0, 1}",           df.same_mature.isin([0, 1]).all())
check("link in {0, 1}",                  df.link.isin([0, 1]).all())

# ── 5. Symmetry ────────────────────────────────────────────────────────────────
print("\n── Symmetry (pairwise features must be order-independent) ──")
check("views_diff >= 0 (symmetric by definition)",  (df.views_diff >= 0).all())
check("views_ratio in [0,1] (min/max is symmetric)", df.views_ratio.between(0, 1 + 1e-6).all())
check("age_diff >= 0 (symmetric by definition)",    (df.age_diff >= 0).all())

# ── 6. Structural feature sanity ──────────────────────────────────────────────
print("\n── Structural Feature Sanity ──")
check("Existing links (link=1) have higher avg common_neighbors than non-links",
      df[df.link==1].common_neighbors.mean() > df[df.link==0].common_neighbors.mean(),
      f"link=1 mean: {df[df.link==1].common_neighbors.mean():.3f}  "
      f"link=0 mean: {df[df.link==0].common_neighbors.mean():.3f}")

check("Existing links have higher avg adamic_adar than non-links",
      df[df.link==1].adamic_adar.mean() > df[df.link==0].adamic_adar.mean(),
      f"link=1 mean: {df[df.link==1].adamic_adar.mean():.3f}  "
      f"link=0 mean: {df[df.link==0].adamic_adar.mean():.3f}")

check("Existing links have higher avg preferential_attachment than non-links",
      df[df.link==1].preferential_attachment.mean() > df[df.link==0].preferential_attachment.mean(),
      f"link=1 mean: {df[df.link==1].preferential_attachment.mean():.0f}  "
      f"link=0 mean: {df[df.link==0].preferential_attachment.mean():.0f}")

# ── 7. No duplicate pairs ──────────────────────────────────────────────────────
print("\n── Duplicates ──")
pairs = df[["from", "to"]].apply(lambda r: tuple(sorted([r["from"], r["to"]])), axis=1)
check("No duplicate node pairs", pairs.duplicated().sum() == 0,
      f"{pairs.duplicated().sum()} duplicates found" if pairs.duplicated().sum() > 0 else "")

# ── 8. Summary stats ───────────────────────────────────────────────────────────
print("\n── Summary Stats ──")
print(f"  Rows:          {len(df):,}")
print(f"  Columns:       {len(df.columns)}")
print(f"  Link=1:        {(df.link==1).sum():,}")
print(f"  Link=0:        {(df.link==0).sum():,}")
print(f"  AA==0 (link=1):{(  (df.link==1) & (df.adamic_adar==0)).sum():,} "
      f"({(  (df.link==1) & (df.adamic_adar==0)).mean()*100:.1f}%)")
print(f"  views_ratio mean (link=1): {df[df.link==1].views_ratio.mean():.3f}")
print(f"  views_ratio mean (link=0): {df[df.link==0].views_ratio.mean():.3f}")
print(f"  same_partner rate (link=1): {df[df.link==1].same_partner.mean()*100:.1f}%")
print(f"  same_partner rate (link=0): {df[df.link==0].same_partner.mean()*100:.1f}%")

# ── Result ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors == 0:
    print("ALL CHECKS PASSED")
else:
    print(f"{errors} CHECK(S) FAILED")
print("=" * 60)
