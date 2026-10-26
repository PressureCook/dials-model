"""Dials pipeline: how many dimensions does the data support, and does a richer
model of environment beat a simple baseline on held out data?

Usage:
    python pipeline.py synthetic
    python pipeline.py ipip path/to/data-final.csv

Needs: numpy pandas scikit-learn
"""
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


# ---------- data ----------
def make_synthetic(n=2000, seed=SEED):
    """Simulate three hidden dials, an environment that nudges them, noisy
    questionnaire items, and a noisy informant rating (the target)."""
    rng = np.random.default_rng(seed)
    adversity = rng.poisson(1.5, n).astype(float)
    warmth = rng.normal(0, 1, n)
    temperament = rng.normal(0, 1, n)
    fear = 0.7 * temperament + rng.normal(0, 0.7, n) - 0.2 * adversity
    reward = rng.normal(0, 1, n) + 0.15 * adversity
    guilt = 0.5 * fear + 0.3 * warmth + rng.normal(0, 0.8, n)
    dials = {"fear": fear, "reward": reward, "guilt": guilt}

    items = {}
    for name, d in dials.items():
        for i in range(5):
            items[f"{name}_{i}"] = 0.7 * d + rng.normal(0, 0.7, n)
    items = pd.DataFrame(items)

    env = pd.DataFrame({"adversity": adversity, "warmth": warmth,
                        "temperament_proxy": temperament + rng.normal(0, 0.5, n)})
    for j in range(10):  # junk columns to show how complex models overfit
        env[f"noise_{j}"] = rng.normal(0, 1, n)
    target = guilt + rng.normal(0, 0.5, n)  # informant rating of the guilt dial
    return items, env, pd.Series(target, name="informant_guilt"), ["adversity", "warmth", "temperament_proxy"]


def load_ipip(path, limit=20000):
    """Open Psychometrics Big Five file (tab separated, items like EXT1..OPN10).
    In that file 0 means missing and age/gender sit in separate columns."""
    df = pd.read_csv(path, sep="\t", nrows=limit)
    cols = [c for c in df.columns if len(c) >= 4 and c[:3].isupper() and c[3:].isdigit()]
    df = df[(df[cols] > 0).all(axis=1) & df["age"].between(10, 90) & df["gender"].isin([1, 2])]
    items = df[cols].reset_index(drop=True)
    env = df[["age", "gender"]].astype(float).reset_index(drop=True)
    return items, env


# ---------- dimensionality ----------
def parallel_analysis(X, n_iter=50, seed=SEED):
    """Keep factors whose eigenvalue beats the 95th percentile of random data."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    real = np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]
    sims = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.normal(size=(n, p)), rowvar=False)))[::-1]
                     for _ in range(n_iter)])
    return int((real > np.percentile(sims, 95, axis=0)).sum()), real


def fit_factors(items, k):
    Z = StandardScaler().fit_transform(items)
    fa = FactorAnalysis(n_components=k, rotation="varimax", random_state=SEED).fit(Z)
    loadings = pd.DataFrame(fa.components_.T, index=items.columns, columns=[f"F{i + 1}" for i in range(k)])
    return fa, loadings, fa.transform(Z)


# ---------- prediction ----------
def compare_models(env, y, baseline_cols):
    cv = KFold(5, shuffle=True, random_state=SEED)
    models = {
        "mean only": (DummyRegressor(), list(env.columns)),
        "simple (few features)": (make_pipeline(StandardScaler(), RidgeCV()), baseline_cols),
        "complex (all features)": (GradientBoostingRegressor(random_state=SEED), list(env.columns)),
    }
    rows = []
    for name, (m, cols) in models.items():
        s = cross_val_score(m, env[cols], y, cv=cv, scoring="r2")
        rows.append((name, s.mean(), s.std()))
    return pd.DataFrame(rows, columns=["model", "R2 (held out)", "spread"]).round(3)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    if mode == "synthetic":
        items, env, y, base = make_synthetic()
    elif mode == "ipip" and len(sys.argv) > 2:
        items, env = load_ipip(sys.argv[2])
        base = list(env.columns)
    else:
        sys.exit(__doc__)

    k, eig = parallel_analysis(items.values)
    print(f"Rows: {len(items)}  Items: {items.shape[1]}")
    print(f"Parallel analysis suggests {k} factors. Top eigenvalues: {np.round(eig[:8], 2)}")
    fa, loadings, scores = fit_factors(items, k)
    for f in loadings.columns:
        top = loadings[f].abs().sort_values(ascending=False).head(5).index.tolist()
        print(f"{f} strongest items: {top}")

    if mode == "ipip":
        y = pd.Series(scores[:, 0], name="factor1_score")
        print("\nTarget: score on factor 1 (a stand in, since this dataset has no upbringing data).")
    else:
        print("\nTarget: noisy informant rating of the hidden guilt dial.")
    print(compare_models(env, y, base).to_string(index=False))
    print("\nRead it like this: if the complex model does not beat the simple one on held out data,"
          "\nthe extra complexity is fitting noise, not finding structure.")


if __name__ == "__main__":
    main()
