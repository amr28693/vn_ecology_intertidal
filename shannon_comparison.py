#!/usr/bin/env python3
"""
Quick Shannon H' comparison.
Run alongside intertidal_vn_entropy.py (same working directory).
Takes ~ seconds.
"""
import numpy as np
import pandas as pd

RAW_DATA = 'Succession_sampling_96-02_data.txt'
SPECIES_6 = ['LL', 'LO', 'NL', 'TT', 'ME', 'MM']

df = pd.read_csv(RAW_DATA, sep='\t')
df.columns = df.columns.str.strip()
df.replace(-999.9, np.nan, inplace=True)
df['Year'] = df['Yr_code'].str.extract(r'(\d{4})').astype(int)

years = sorted(df['Year'].unique())

print("Year   Shannon_H   Simpson_D   VNE_S_norm")
print("-" * 48)

shannon_vals = []
simpson_vals = []

for yr in years:
    sub = df[df['Year'] == yr][SPECIES_6].dropna(how='any')
    # Pool abundances across all sites for this year
    totals = sub.sum(axis=0).values.astype(float)
    totals = totals[totals > 0]
    p = totals / totals.sum()

    H = -np.sum(p * np.log(p))  # Shannon H' (nats)
    D = 1 - np.sum(p ** 2)      # Simpson's D

    # VNE for comparison
    R = sub.corr(method='spearman')
    rho = R.values / np.trace(R.values)
    eigs = np.linalg.eigvalsh(rho)
    eigs = np.clip(eigs, 1e-15, None)
    S = -np.sum(eigs * np.log(eigs)) / np.log(2)
    S_norm = S / np.log2(6)

    shannon_vals.append(H)
    simpson_vals.append(D)
    print(f"{yr}   {H:.4f}       {D:.4f}       {S_norm:.4f}")

# Linear trends
yr_arr = np.array(years, dtype=float)
h_slope, h_int = np.polyfit(yr_arr, shannon_vals, 1)
d_slope, d_int = np.polyfit(yr_arr, simpson_vals, 1)

print(f"\nShannon slope: {h_slope:.6f}/yr  (P via permutation below)")
print(f"Simpson slope: {d_slope:.6f}/yr")

# Quick permutation test for Shannon slope
rng = np.random.default_rng(42)
null_slopes = []
for _ in range(10000):
    perm = rng.permutation(shannon_vals)
    null_slopes.append(np.polyfit(yr_arr, perm, 1)[0])
p_shannon = np.mean(np.array(null_slopes) <= h_slope)
print(f"Shannon slope P (one-tailed, decline): {p_shannon:.4f}")

null_slopes_d = []
for _ in range(10000):
    perm = rng.permutation(simpson_vals)
    null_slopes_d.append(np.polyfit(yr_arr, perm, 1)[0])
p_simpson = np.mean(np.array(null_slopes_d) <= d_slope)
print(f"Simpson slope P (one-tailed, decline): {p_simpson:.4f}")
