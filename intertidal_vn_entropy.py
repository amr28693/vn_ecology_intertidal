#!/usr/bin/env python3
"""
================================================================================
Information Geometry of Intertidal Community Organization:
Von Neumann Entropy Detects Alternative Stable State Architecture
in Gulf of Maine Succession
================================================================================

Single-script analysis and validation.

Data:    Petraitis, P. S. & Vidargas, N. (2006). Ecology, 87(3):796.
         Figshare DOI: 10.6084/m9.figshare.3526004
         File: Succession_sampling_96-02_data.txt (588 rows, unmodified)

Method:  Compute species correlation density matrices per stratum.
         Extract Von Neumann entropy as a measure of community
         interaction dimensionality. Detect
         data-driven community partitions via hierarchical clustering
         on correlation distance.

Usage:   python intertidal_vn_entropy.py
         (expects Succession_sampling_96-02_data.txt in working dir)

Output:  intertidal_vn_output/
           results/     — CSV tables
           figures/     — publication figures
           validation/  — bootstrap, permutation, summary

Author:  Anderson M. Rodriguez (2026)
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import os, sys, json, warnings, time
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_DATA = 'Succession_sampling_96-02_data.txt'
BASE_DIR = 'intertidal_vn_output'
RES_DIR  = f'{BASE_DIR}/results'
FIG_DIR  = f'{BASE_DIR}/figures'
VAL_DIR  = f'{BASE_DIR}/validation'

SPECIES_8 = ['LL', 'LO', 'NL', 'TT', 'FV', 'AN', 'ME', 'MM']
SPECIES_6 = ['LL', 'LO', 'NL', 'TT', 'ME', 'MM']  # complete all 7 years

SPECIES_NAMES = {
    'LL': 'Littorina littorea',   'LO': 'Littorina obtusata',
    'NL': 'Nucella lapillus',     'TT': 'Tectura testudinalis',
    'FV': 'Fucus vesiculosus',    'AN': 'Ascophyllum nodosum',
    'ME': 'Mytilus edulis',       'MM': 'Modiolus modiolus',
}

GUILDS = {
    'Grazers':    ['LL', 'LO', 'NL', 'TT'],
    'Macroalgae': ['FV', 'AN'],
    'Bivalves':   ['ME', 'MM'],
}

N_BOOT = 1000
N_PERM = 1000
ALPHA  = 0.05
RNG_SEED = 42

for d in [BASE_DIR, RES_DIR, FIG_DIR, VAL_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def vn_entropy(corr_matrix, base=2):
    """
    Von Neumann entropy of density matrix derived from correlation matrix.

    Given correlation matrix R (positive semidefinite, diagonal = 1),
    form density matrix rho = R / tr(R), then:
        S(rho) = -tr(rho log rho) = -sum_i lambda_i log(lambda_i)
    where lambda_i are eigenvalues of rho.

    Interpretation: effective dimensionality of species interaction
    network. S = 0 when one eigenvalue dominates (all species perfectly
    correlated, 1D structure). S = log2(n) when all eigenvalues equal
    (species uncorrelated, maximum-dimensional structure).
    """
    R = corr_matrix.values if hasattr(corr_matrix, 'values') else np.array(corr_matrix)
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        return np.nan
    tr = np.trace(R)
    if tr < 1e-10:
        return np.nan
    rho = R / tr
    try:
        eigs = np.linalg.eigvalsh(rho)
    except np.linalg.LinAlgError:
        return np.nan
    eigs = np.clip(eigs, 1e-15, None)
    return -np.sum(eigs * np.log(eigs)) / np.log(base)


def detect_communities(corr_matrix, sp_labels, k=3):
    """
    Data-driven community detection via Ward hierarchical clustering
    on correlation distance (1 - |r|).
    """
    R = corr_matrix.values if hasattr(corr_matrix, 'values') else np.array(corr_matrix)
    dist = 1 - np.abs(R)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=0.0)
    dist = np.clip(dist, 0, 2)
    condensed = squareform(dist, checks=False)
    if not np.all(np.isfinite(condensed)):
        return {sp: 1 for sp in sp_labels}
    Z = linkage(condensed, method='ward')
    clusters = fcluster(Z, t=k, criterion='maxclust')
    return dict(zip(sp_labels, clusters.tolist()))


def guild_match_score(detected, sp_labels, a_priori_guilds):
    """
    Pairwise agreement between data-driven and a priori groupings.
    Returns fraction of species pairs where both methods agree
    on same-group vs different-group membership.
    """
    n = len(sp_labels)
    ap = {}
    for gn, spp in a_priori_guilds.items():
        for s in spp:
            ap[s] = gn
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = sp_labels[i], sp_labels[j]
            if (detected[si] == detected[sj]) == (ap.get(si) == ap.get(sj)):
                agree += 1
            total += 1
    return agree / total if total > 0 else 0


def vn_from_df(df_sub, sp_list, method='spearman'):
    """Convenience: compute VN entropy from a dataframe subset."""
    sub = df_sub[sp_list].dropna(how='any')
    if len(sub) < len(sp_list):
        return np.nan
    R = sub.corr(method=method)
    return vn_entropy(R)


# ============================================================================
# ANALYSIS
# ============================================================================

def load_data():
    """Load raw data. Only transformation: -999.9 -> NaN."""
    print("Loading data...")
    df = pd.read_csv(RAW_DATA, sep='\t')
    df.columns = df.columns.str.strip()
    df.replace(-999.9, np.nan, inplace=True)
    df['Year']   = df['Yr_code'].str.extract(r'(\d{4})').astype(int)
    df['Season'] = df['Yr_code'].str.extract(r'\d{4}([ab])')
    assert len(df) == 588, f"Expected 588 rows, got {len(df)}"
    print(f"  {len(df)} rows | {df['Year'].nunique()} years ({df['Year'].min()}-{df['Year'].max()}) | "
          f"{df['Bay'].nunique()} bays | {df['Site'].nunique()} sites | "
          f"sizes: {sorted(df['Size'].unique())}")
    return df


def analyze(df):
    """
    Compute Von Neumann entropy and community structure
    across all strata: year, size, year x size, bay.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: Von Neumann Entropy of Species Correlation Matrices")
    print("=" * 70)

    years = sorted(df['Year'].unique())
    sizes = sorted(df['Size'].unique())
    bays  = sorted(df['Bay'].unique())
    sites = sorted(df['Site'].unique())
    results = []

    # --- (1) Temporal: 6-species, all 7 years ---
    print("\n(1) TEMPORAL — 6-species, 7 years")
    print(f"    {'Year':>4}  {'S_vn':>7}  {'S/Smax':>7}  {'Match':>6}  {'n':>4}  Clusters")
    for yr in years:
        sub = df[df['Year'] == yr]
        complete = sub[SPECIES_6].dropna(how='any')
        if len(complete) < len(SPECIES_6):
            continue
        R = complete.corr(method='spearman')
        S = vn_entropy(R)
        cl = detect_communities(R, SPECIES_6, k=2)
        ms = guild_match_score(cl, SPECIES_6,
                               {'Grazers': ['LL','LO','NL','TT'], 'Bivalves': ['ME','MM']})
        Smax = np.log2(6)
        print(f"    {yr:>4}  {S:>7.4f}  {S/Smax:>7.4f}  {ms:>6.2f}  {len(complete):>4}  {cl}")
        results.append(dict(stratum='year_6sp', year=yr, size='all', bay='all',
                            n_species=6, n_obs=len(complete),
                            S_vn=S, S_norm=S/Smax, guild_match=ms,
                            clusters=json.dumps(cl)))

    # --- (2) Temporal: 8-species, 1998-2002 ---
    print("\n(2) TEMPORAL — 8-species, 1998-2002")
    print(f"    {'Year':>4}  {'S_vn':>7}  {'S/Smax':>7}  {'Match':>6}  {'n':>4}  Clusters")
    for yr in years:
        sub = df[df['Year'] == yr]
        complete = sub[SPECIES_8].dropna(how='any')
        if len(complete) < len(SPECIES_8):
            continue
        R = complete.corr(method='spearman')
        S = vn_entropy(R)
        cl = detect_communities(R, SPECIES_8, k=3)
        ms = guild_match_score(cl, SPECIES_8, GUILDS)
        Smax = np.log2(8)
        print(f"    {yr:>4}  {S:>7.4f}  {S/Smax:>7.4f}  {ms:>6.2f}  {len(complete):>4}  {cl}")
        results.append(dict(stratum='year_8sp', year=yr, size='all', bay='all',
                            n_species=8, n_obs=len(complete),
                            S_vn=S, S_norm=S/Smax, guild_match=ms,
                            clusters=json.dumps(cl)))

    # --- (3) Spatial: 8-species by clearing size ---
    print("\n(3) SPATIAL — 8-species, by quadrat size (pooled years)")
    print(f"    {'Size':>4}  {'S_vn':>7}  {'S/Smax':>7}  {'Match':>6}  {'n':>4}  Clusters")
    for sz in sizes:
        sub = df[df['Size'] == sz]
        complete = sub[SPECIES_8].dropna(how='any')
        if len(complete) < len(SPECIES_8):
            continue
        R = complete.corr(method='spearman')
        S = vn_entropy(R)
        cl = detect_communities(R, SPECIES_8, k=3)
        ms = guild_match_score(cl, SPECIES_8, GUILDS)
        Smax = np.log2(8)
        print(f"    {sz:>4}  {S:>7.4f}  {S/Smax:>7.4f}  {ms:>6.2f}  {len(complete):>4}  {cl}")
        results.append(dict(stratum='size_8sp', year='all', size=sz, bay='all',
                            n_species=8, n_obs=len(complete),
                            S_vn=S, S_norm=S/Smax, guild_match=ms,
                            clusters=json.dumps(cl)))

    # --- (4) Year x Size: 6-species (n >= 8 filter) ---
    print("\n(4) YEAR x SIZE — 6-species (n >= 8)")
    yr_sz_matrix = {}
    for yr in years:
        row = {}
        for sz in sizes:
            sub = df[(df['Year'] == yr) & (df['Size'] == sz)]
            complete = sub[SPECIES_6].dropna(how='any')
            if len(complete) >= 8:
                R = complete.corr(method='spearman')
                S = vn_entropy(R)
                if not np.isnan(S):
                    row[sz] = S
                    results.append(dict(stratum='year_size_6sp', year=yr, size=sz,
                                        bay='all', n_species=6, n_obs=len(complete),
                                        S_vn=S, S_norm=S/np.log2(6)))
        if row:
            yr_sz_matrix[yr] = row
    if yr_sz_matrix:
        ys_df = pd.DataFrame(yr_sz_matrix).T
        ys_df.index.name = 'Year'
        ys_df.columns = [f'Size_{c}' for c in ys_df.columns]
        print(ys_df.round(4).to_string())

    # --- (5) Bay-level: 8-species ---
    print("\n(5) BAY — 8-species (pooled years)")
    for bay in bays:
        sub = df[df['Bay'] == bay]
        complete = sub[SPECIES_8].dropna(how='any')
        if len(complete) >= len(SPECIES_8):
            R = complete.corr(method='spearman')
            S = vn_entropy(R)
            cl = detect_communities(R, SPECIES_8, k=3)
            ms = guild_match_score(cl, SPECIES_8, GUILDS)
            Smax = np.log2(8)
            S_str = f"{S:.4f}" if not np.isnan(S) else "  NaN "
            print(f"    {bay}: S_vn={S_str}  match={ms:.2f}  {cl}")
            results.append(dict(stratum=f'bay_{bay}', year='all', size='all',
                                bay=bay, n_species=8, n_obs=len(complete),
                                S_vn=S, S_norm=S/Smax if not np.isnan(S) else np.nan,
                                guild_match=ms, clusters=json.dumps(cl)))

    # --- (6) Bay x Year: 6-species ---
    print("\n(6) BAY x YEAR — 6-species")
    bay_yr = {}
    for bay in bays:
        row = {}
        for yr in years:
            sub = df[(df['Bay'] == bay) & (df['Year'] == yr)]
            complete = sub[SPECIES_6].dropna(how='any')
            if len(complete) >= 6:
                S = vn_from_df(sub, SPECIES_6)
                if not np.isnan(S):
                    row[yr] = S
                    results.append(dict(stratum='bay_year_6sp', year=yr, size='all',
                                        bay=bay, n_species=6, n_obs=len(complete),
                                        S_vn=S, S_norm=S/np.log2(6)))
        if row:
            bay_yr[bay] = row
    if bay_yr:
        by_df = pd.DataFrame(bay_yr).T
        by_df.index.name = 'Bay'
        print(by_df.round(4).to_string())

    # Save
    rdf = pd.DataFrame(results)
    rdf.to_csv(f'{RES_DIR}/vn_entropy_all_strata.csv', index=False)
    print(f"\n  Saved {len(rdf)} results to {RES_DIR}/vn_entropy_all_strata.csv")
    return rdf


# ============================================================================
# VALIDATION
# ============================================================================

def validate(df, results_df):
    """
    V1: Bootstrap CIs (site-resampled)
    V2: Permutation null (species-label shuffling)
    V3: Permutation null (row shuffling — trajectory significance)
    """
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    rng = np.random.default_rng(RNG_SEED)
    years = sorted(df['Year'].unique())
    sizes = sorted(df['Size'].unique())
    sites = sorted(df['Site'].unique())
    val_results = []

    # =========================
    # V1: BOOTSTRAP
    # =========================
    print("\n--- V1: Bootstrap Confidence Intervals (site-resampled) ---")

    # Temporal CIs
    print(f"\n  TEMPORAL (6-species)")
    print(f"  {'Year':>4}  {'Observed':>9}  {'CI_lo':>9}  {'CI_hi':>9}  {'SE':>7}")
    for yr in years:
        yr_data = df[df['Year'] == yr]
        yr_sites = yr_data['Site'].unique()
        obs = vn_from_df(yr_data, SPECIES_6)

        boot = []
        for _ in range(N_BOOT):
            ss = rng.choice(yr_sites, size=len(yr_sites), replace=True)
            bd = pd.concat([yr_data[yr_data['Site'] == s] for s in ss], ignore_index=True)
            bv = vn_from_df(bd, SPECIES_6)
            if not np.isnan(bv):
                boot.append(bv)

        if len(boot) > 50:
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            se = np.std(boot)
            print(f"  {yr:>4}  {obs:>9.4f}  {ci_lo:>9.4f}  {ci_hi:>9.4f}  {se:>7.4f}")
            val_results.append(dict(test='V1_bootstrap', stratum='year_6sp',
                                    level=yr, observed=obs, ci_lo=ci_lo, ci_hi=ci_hi, se=se))

    # Temporal slope bootstrap
    obs_vals = np.array([vn_from_df(df[df['Year'] == yr], SPECIES_6) for yr in years])
    yr_arr = np.array(years, dtype=float)
    mask = ~np.isnan(obs_vals)
    obs_slope = np.polyfit(yr_arr[mask], obs_vals[mask], 1)[0]

    boot_slopes = []
    for _ in range(N_BOOT):
        ss = rng.choice(sites, size=len(sites), replace=True)
        bd = pd.concat([df[df['Site'] == s] for s in ss], ignore_index=True)
        bvals = np.array([vn_from_df(bd[bd['Year'] == yr], SPECIES_6) for yr in years])
        m = ~np.isnan(bvals)
        if m.sum() >= 3:
            boot_slopes.append(np.polyfit(yr_arr[m], bvals[m], 1)[0])

    slope_sig = False
    if len(boot_slopes) > 50:
        s_lo, s_hi = np.percentile(boot_slopes, [2.5, 97.5])
        frac_neg = np.mean(np.array(boot_slopes) < 0)
        slope_sig = s_hi < 0
        print(f"\n  Temporal slope: {obs_slope:.6f}/yr  95% CI: [{s_lo:.6f}, {s_hi:.6f}]")
        print(f"  Fraction negative: {frac_neg:.3f}  Significant decline: {'YES' if slope_sig else 'NO'}")
        val_results.append(dict(test='V1_slope', stratum='temporal', level='slope',
                                observed=obs_slope, ci_lo=s_lo, ci_hi=s_hi, frac_negative=frac_neg))

    # Spatial CIs
    print(f"\n  SPATIAL (8-species)")
    print(f"  {'Size':>4}  {'Observed':>9}  {'CI_lo':>9}  {'CI_hi':>9}  {'SE':>7}")
    for sz in sizes:
        sz_data = df[df['Size'] == sz]
        sz_sites = sz_data['Site'].unique()
        obs = vn_from_df(sz_data, SPECIES_8)

        boot = []
        for _ in range(N_BOOT):
            ss = rng.choice(sz_sites, size=len(sz_sites), replace=True)
            bd = pd.concat([sz_data[sz_data['Site'] == s] for s in ss], ignore_index=True)
            bv = vn_from_df(bd, SPECIES_8)
            if not np.isnan(bv):
                boot.append(bv)

        if len(boot) > 50:
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            se = np.std(boot)
            print(f"  {sz:>4}  {obs:>9.4f}  {ci_lo:>9.4f}  {ci_hi:>9.4f}  {se:>7.4f}")
            val_results.append(dict(test='V1_bootstrap', stratum='size_8sp',
                                    level=sz, observed=obs, ci_lo=ci_lo, ci_hi=ci_hi, se=se))

    # Size 0 vs Size 2 pairwise
    diffs = []
    for _ in range(N_BOOT):
        ss = rng.choice(sites, size=len(sites), replace=True)
        bd = pd.concat([df[df['Site'] == s] for s in ss], ignore_index=True)
        v0 = vn_from_df(bd[bd['Size'] == 0], SPECIES_8)
        v2 = vn_from_df(bd[bd['Size'] == 2], SPECIES_8)
        if not np.isnan(v0) and not np.isnan(v2):
            diffs.append(v0 - v2)

    size_sig = False
    if len(diffs) > 50:
        obs_diff = vn_from_df(df[df['Size'] == 0], SPECIES_8) - \
                   vn_from_df(df[df['Size'] == 2], SPECIES_8)
        d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
        frac_pos = np.mean(np.array(diffs) > 0)
        size_sig = d_lo > 0
        print(f"\n  Size 0 vs Size 2: diff={obs_diff:.4f}  95% CI: [{d_lo:.4f}, {d_hi:.4f}]")
        print(f"  Fraction > 0: {frac_pos:.3f}  Significant: {'YES' if size_sig else 'NO'}")
        val_results.append(dict(test='V1_pairwise', stratum='size_0v2', level='diff',
                                observed=obs_diff, ci_lo=d_lo, ci_hi=d_hi, frac_positive=frac_pos))

    # =========================
    # V2: SPECIES PERMUTATION
    # =========================
    print("\n--- V2: Permutation Test — Species Label Shuffling ---")
    print("  (Null: shuffle species labels within each row)")

    print(f"\n  TEMPORAL (6-species)")
    for yr in years:
        yr_data = df[df['Year'] == yr].copy()
        obs = vn_from_df(yr_data, SPECIES_6)
        null_dist = []
        for _ in range(N_PERM):
            perm = yr_data.copy()
            for idx in perm.index:
                vals = perm.loc[idx, SPECIES_6].values.copy()
                rng.shuffle(vals)
                perm.loc[idx, SPECIES_6] = vals
            nv = vn_from_df(perm, SPECIES_6)
            if not np.isnan(nv):
                null_dist.append(nv)
        if len(null_dist) > 50:
            nm, ns = np.mean(null_dist), np.std(null_dist)
            z = (obs - nm) / ns if ns > 1e-10 else np.nan
            p = np.mean(np.array(null_dist) <= obs)
            print(f"  {yr}: obs={obs:.4f}  null={nm:.4f}+/-{ns:.4f}  z={z:+.2f}  p={p:.4f}")
            val_results.append(dict(test='V2_species_perm', stratum='year_6sp',
                                    level=yr, observed=obs, null_mean=nm, null_std=ns,
                                    z_score=z, p_value=p))

    print(f"\n  SPATIAL (8-species)")
    for sz in sizes:
        sz_data = df[df['Size'] == sz].copy()
        obs = vn_from_df(sz_data, SPECIES_8)
        if np.isnan(obs):
            continue
        null_dist = []
        for _ in range(N_PERM):
            perm = sz_data.copy()
            cmask = perm[SPECIES_8].notna().all(axis=1)
            for idx in perm[cmask].index:
                vals = perm.loc[idx, SPECIES_8].values.copy()
                rng.shuffle(vals)
                perm.loc[idx, SPECIES_8] = vals
            nv = vn_from_df(perm, SPECIES_8)
            if not np.isnan(nv):
                null_dist.append(nv)
        if len(null_dist) > 50:
            nm, ns = np.mean(null_dist), np.std(null_dist)
            z = (obs - nm) / ns if ns > 1e-10 else np.nan
            p = np.mean(np.array(null_dist) <= obs)
            print(f"  Size {sz}: obs={obs:.4f}  null={nm:.4f}+/-{ns:.4f}  z={z:+.2f}  p={p:.4f}")
            val_results.append(dict(test='V2_species_perm', stratum='size_8sp',
                                    level=sz, observed=obs, null_mean=nm, null_std=ns,
                                    z_score=z, p_value=p))

    # =========================
    # V3: ROW PERMUTATION
    # =========================
    print("\n--- V3: Permutation Test — Row Shuffling (trajectory significance) ---")
    print("  (Null: shuffle year/size labels across rows)")

    obs_traj = [vn_from_df(df[df['Year'] == yr], SPECIES_6) for yr in years]
    obs_range = np.nanmax(obs_traj) - np.nanmin(obs_traj)
    clean = [(y, v) for y, v in zip(range(len(years)), obs_traj) if not np.isnan(v)]
    obs_slope_v3 = np.polyfit([c[0] for c in clean], [c[1] for c in clean], 1)[0]

    null_ranges, null_slopes = [], []
    for _ in range(N_PERM):
        perm = df.copy()
        perm['Year'] = rng.permutation(perm['Year'].values)
        pt = [vn_from_df(perm[perm['Year'] == yr], SPECIES_6) for yr in years]
        pc = [(y, v) for y, v in zip(range(len(years)), pt) if not np.isnan(v)]
        if len(pc) >= 3:
            null_ranges.append(np.nanmax(pt) - np.nanmin(pt))
            null_slopes.append(np.polyfit([c[0] for c in pc], [c[1] for c in pc], 1)[0])

    if len(null_ranges) > 50:
        p_range = np.mean(np.array(null_ranges) >= obs_range)
        p_slope = np.mean(np.array(null_slopes) <= obs_slope_v3)
        print(f"  Temporal range: {obs_range:.4f}  p={p_range:.4f}")
        print(f"  Temporal slope: {obs_slope_v3:.6f}  p={p_slope:.4f}")
        val_results.append(dict(test='V3_row_perm', stratum='temporal', level='range',
                                observed=obs_range, p_value=p_range))
        val_results.append(dict(test='V3_row_perm', stratum='temporal', level='slope',
                                observed=obs_slope_v3, p_value=p_slope))

    obs_sp = [vn_from_df(df[df['Size'] == sz], SPECIES_8) for sz in sizes]
    obs_sp_range = np.nanmax(obs_sp) - np.nanmin(obs_sp)
    null_sp_ranges = []
    for _ in range(N_PERM):
        perm = df.copy()
        perm['Size'] = rng.permutation(perm['Size'].values)
        ps = [vn_from_df(perm[perm['Size'] == sz], SPECIES_8) for sz in sizes]
        null_sp_ranges.append(np.nanmax(ps) - np.nanmin(ps))

    if len(null_sp_ranges) > 50:
        p_sp = np.mean(np.array(null_sp_ranges) >= obs_sp_range)
        print(f"  Spatial range: {obs_sp_range:.4f}  p={p_sp:.4f}")
        val_results.append(dict(test='V3_row_perm', stratum='spatial', level='range',
                                observed=obs_sp_range, p_value=p_sp))

    # Save
    vdf = pd.DataFrame(val_results)
    vdf.to_csv(f'{VAL_DIR}/validation_results.csv', index=False)
    return vdf


# ============================================================================
# FIGURES
# ============================================================================

def make_figures(df, results_df, val_df):
    """Publication-quality figures."""
    print("\n" + "=" * 70)
    print("FIGURES")
    print("=" * 70)

    yr6 = results_df[results_df['stratum'] == 'year_6sp'].copy()
    yr6['year'] = pd.to_numeric(yr6['year'], errors='coerce')
    yr6 = yr6.sort_values('year')
    yr8 = results_df[results_df['stratum'] == 'year_8sp'].copy()
    yr8['year'] = pd.to_numeric(yr8['year'], errors='coerce')
    yr8 = yr8.sort_values('year')
    sz8 = results_df[results_df['stratum'] == 'size_8sp'].copy()
    sz8['size'] = pd.to_numeric(sz8['size'], errors='coerce')
    sz8 = sz8.sort_values('size')

    # Bootstrap CIs for error bars
    boot_yr = val_df[(val_df['test'] == 'V1_bootstrap') & (val_df['stratum'] == 'year_6sp')].copy()
    boot_sz = val_df[(val_df['test'] == 'V1_bootstrap') & (val_df['stratum'] == 'size_8sp')].copy()

    # ========================
    # FIGURE 1: Main result (2 panels)
    # ========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Temporal
    if len(yr6) > 0 and len(boot_yr) > 0:
        merged = yr6.merge(boot_yr[['level', 'ci_lo', 'ci_hi']], left_on='year', right_on='level', how='left')
        yerr_lo = merged['S_norm'].values - merged['ci_lo'].values / np.log2(6)
        yerr_hi = merged['ci_hi'].values / np.log2(6) - merged['S_norm'].values
        yerr = np.array([np.abs(yerr_lo), np.abs(yerr_hi)])
        ax1.errorbar(merged['year'], merged['S_norm'], yerr=yerr,
                     fmt='o-', color='#C62828', lw=2.5, ms=8, capsize=4, capthick=1.5,
                     label='6-species (complete)', zorder=5)
    elif len(yr6) > 0:
        ax1.plot(yr6['year'], yr6['S_norm'], 'o-', color='#C62828', lw=2.5, ms=8, label='6-species')

    if len(yr8) > 0:
        ax1.plot(yr8['year'], yr8['S_norm'], 's--', color='#1565C0', lw=1.8, ms=6,
                 alpha=0.7, label='8-species (1998+)')

    # Trend line
    if len(yr6) > 0:
        x = yr6['year'].values.astype(float)
        y = yr6['S_norm'].values
        z = np.polyfit(x, y, 1)
        ax1.plot(x, np.polyval(z, x), ':', color='#C62828', alpha=0.4, lw=1.5)

    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Normalized Von Neumann Entropy  $S / S_{max}$', fontsize=11)
    ax1.set_title('A.  Community Self-Organization Over Succession', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(alpha=0.2)
    ax1.set_ylim(0.82, 0.98)

    # Panel B: Spatial
    if len(sz8) > 0:
        if len(boot_sz) > 0:
            merged = sz8.merge(boot_sz[['level', 'ci_lo', 'ci_hi']], left_on='size', right_on='level', how='left')
            yerr_lo = merged['S_norm'].values - merged['ci_lo'].values / np.log2(8)
            yerr_hi = merged['ci_hi'].values / np.log2(8) - merged['S_norm'].values
            yerr = np.array([np.abs(yerr_lo), np.abs(yerr_hi)])
            ax2.errorbar(merged['size'], merged['S_norm'], yerr=yerr,
                         fmt='D-', color='#2E7D32', lw=2.5, ms=9, capsize=4, capthick=1.5, zorder=5)
        else:
            ax2.plot(sz8['size'], sz8['S_norm'], 'D-', color='#2E7D32', lw=2.5, ms=9)

        # fill removed — zoomed y-axis makes fill misleading

    # Mark Size=2 minimum
    if len(sz8) > 0:
        min_idx = sz8['S_norm'].idxmin()
        min_row = sz8.loc[min_idx]
        ax2.annotate(f'Size {int(min_row["size"])} m²\n(most organized)',
                     xy=(min_row['size'], min_row['S_norm']),
                     xytext=(min_row['size'] + 2, min_row['S_norm'] - 0.015),
                     fontsize=9, ha='center',
                     arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

    ax2.set_xlabel('Quadrat Clearing Size (m²)', fontsize=12)
    ax2.set_ylabel('Normalized Von Neumann Entropy  $S / S_{max}$', fontsize=11)
    ax2.set_title('B.  Interaction Structure Along Size Gradient', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.2)
    ax2.set_ylim(0.82, 0.96)

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig1_vn_entropy_main.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIG_DIR}/fig1_vn_entropy_main.pdf', bbox_inches='tight')
    plt.close()
    print("  fig1_vn_entropy_main.png/pdf")

    # ========================
# FIGURE 2: Community structure (guild match only)
    # ========================
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    match_labels = []
    match_vals = []
    match_colors = []
    if len(yr8) > 0 and 'guild_match' in yr8.columns:
        for _, r in yr8.iterrows():
            match_labels.append(f"{int(r['year'])}")
            match_vals.append(r['guild_match'])
            match_colors.append('#E65100')
    if len(sz8) > 0 and 'guild_match' in sz8.columns:
        for _, r in sz8.iterrows():
            match_labels.append(f"Sz{int(r['size'])}")
            match_vals.append(r['guild_match'])
            match_colors.append('#1B5E20')
    if match_vals:
        x_pos = np.arange(len(match_vals))
        ax.bar(x_pos, match_vals, color=match_colors, alpha=0.8, width=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(match_labels, rotation=45, fontsize=8)
        ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
        ax.set_ylim(0.4, 1.05)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='#E65100', label='By Year'),
                           Patch(color='#1B5E20', label='By Size')], fontsize=8)
    ax.set_ylabel('Pairwise Agreement')
    ax.set_title('Data-Driven vs A Priori Community Structure', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig2_community_structure.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIG_DIR}/fig2_community_structure.pdf', bbox_inches='tight')
    plt.close()
    print("  fig2_community_structure.png/pdf")

    # ========================
    # FIGURE 3: Bay replication
    # ========================
    bay_yr = results_df[results_df['stratum'] == 'bay_year_6sp'].copy()
    if len(bay_yr) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        bay_colors = {'BC': '#1565C0', 'MC': '#43A047', 'SC': '#EF6C00', 'TC': '#8E24AA'}
        bay_names = {'BC': 'Birch Cove', 'MC': 'Moose Cove',
                     'SC': 'Sandy Cove', 'TC': 'Timber Cove'}
        for bay in sorted(bay_yr['bay'].unique()):
            bdata = bay_yr[bay_yr['bay'] == bay].sort_values('year')
            ax.plot(bdata['year'], bdata['S_norm'], 'o-',
                    color=bay_colors.get(bay, 'gray'),
                    label=f"{bay} ({bay_names.get(bay, '')})", lw=2, ms=5)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Normalized VN Entropy', fontsize=11)
        ax.set_title('Figure 3: Bay-Level Replication', fontsize=12, fontweight='bold')
        ax.legend(); ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/fig3_bay_replication.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{FIG_DIR}/fig3_bay_replication.pdf', bbox_inches='tight')
        plt.close()
        print("  fig3_bay_replication.png/pdf")

    # ========================
    # FIGURE S1: Year x Size heatmap
    # ========================
    yr_sz = results_df[results_df['stratum'] == 'year_size_6sp'].copy()
    if len(yr_sz) > 0:
        pivot = yr_sz.pivot(index='year', columns='size', values='S_norm')
        vals = pivot.values.astype(float)
        fig, ax = plt.subplots(figsize=(8, 5))
        valid_vals = vals[~np.isnan(vals)]
        im = ax.imshow(vals, aspect='auto', cmap='RdYlGn_r',
                       vmin=valid_vals.min() - 0.02,
                       vmax=valid_vals.max() + 0.02)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'Size {int(c)}' for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([int(y) for y in pivot.index])
        ax.set_xlabel('Quadrat Size (m²)'); ax.set_ylabel('Year')
        ax.set_title('Figure S1: VN Entropy (Year × Size)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='S / Smax')
        mean_val = valid_vals.mean()
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8,
                            color='white' if v < mean_val else 'black')
                else:
                    ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='gray')
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/figS1_year_size_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  figS1_year_size_heatmap.png")


# ============================================================================
# SUMMARY
# ============================================================================

def summary(results_df, val_df):
    """Print and save final summary."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    lines = []
    lines.append("=" * 60)
    lines.append("INFORMATION GEOMETRY OF INTERTIDAL COMMUNITY ORGANIZATION")
    lines.append("Von Neumann Entropy Analysis — Results Summary")
    lines.append("=" * 60)

    # Temporal
    yr6 = results_df[results_df['stratum'] == 'year_6sp'].sort_values('year')
    if len(yr6) > 0:
        lines.append("")
        lines.append("TEMPORAL DECLINE (6-species, 1996-2002):")
        lines.append(f"  Start: {yr6.iloc[0]['S_norm']:.4f}  End: {yr6.iloc[-1]['S_norm']:.4f}")
        lines.append(f"  Change: {yr6.iloc[-1]['S_norm'] - yr6.iloc[0]['S_norm']:.4f} "
                     f"({100*(yr6.iloc[-1]['S_norm'] - yr6.iloc[0]['S_norm'])/yr6.iloc[0]['S_norm']:.1f}%)")

    # Spatial
    sz8 = results_df[results_df['stratum'] == 'size_8sp'].sort_values('size')
    if len(sz8) > 0:
        min_row = sz8.loc[sz8['S_norm'].idxmin()]
        max_row = sz8.loc[sz8['S_norm'].idxmax()]
        lines.append("")
        lines.append("CLEARING-SIZE GRADIENT (8-species, pooled):")
        lines.append(f"  Most organized: Size {int(min_row['size'])} m² (S/Smax = {min_row['S_norm']:.4f})")
        lines.append(f"  Least organized: Size {int(max_row['size'])} m² (S/Smax = {max_row['S_norm']:.4f})")
        lines.append(f"  Range: {max_row['S_norm'] - min_row['S_norm']:.4f}")

    # Validation
    if val_df is not None and len(val_df) > 0:
        lines.append("")
        lines.append("VALIDATION:")

        slope = val_df[(val_df['test'] == 'V1_slope')]
        if len(slope) > 0:
            sr = slope.iloc[0]
            sig = "YES" if sr.get('ci_hi', 1) < 0 else "NO"
            lines.append(f"  Temporal decline significant (bootstrap): {sig}")
            lines.append(f"    Slope 95% CI: [{sr.get('ci_lo', 'N/A'):.6f}, {sr.get('ci_hi', 'N/A'):.6f}]")

        pw = val_df[val_df['test'] == 'V1_pairwise']
        if len(pw) > 0:
            pr = pw.iloc[0]
            sig = "YES" if pr.get('ci_lo', -1) > 0 else "NO"
            lines.append(f"  Size 0 > Size 2 significant (bootstrap): {sig}")

        v2 = val_df[val_df['test'] == 'V2_species_perm']
        if len(v2) > 0:
            n_sig = (v2['p_value'] < 0.05).sum()
            lines.append(f"  Species permutation: {n_sig}/{len(v2)} strata significant at p<0.05")

        v3 = val_df[val_df['test'] == 'V3_row_perm']
        if len(v3) > 0:
            for _, r in v3.iterrows():
                lines.append(f"  Row permutation ({r['stratum']} {r['level']}): p={r['p_value']:.4f}")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    with open(f'{VAL_DIR}/summary.txt', 'w') as f:
        f.write(report)

    return report


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    t0 = time.time()

    if not os.path.exists(RAW_DATA):
        print(f"ERROR: {RAW_DATA} not found in working directory.")
        print(f"Download from: https://figshare.com/articles/dataset/3526004")
        sys.exit(1)

    df = load_data()
    results_df = analyze(df)
    val_df = validate(df, results_df)
    make_figures(df, results_df, val_df)
    summary(results_df, val_df)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"All outputs in: {BASE_DIR}/")
