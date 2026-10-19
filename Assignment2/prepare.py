"""Inspect only raw data, moments and ranks before any model/risk fitting."""
from pathlib import Path
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'assignment2-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
TAB = ROOT / 'tables'
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 140, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False})

def read(problem):
    return pd.read_csv(ROOT / 'data' / f'problem{problem}.csv')

def moments(data):
    """Descriptive central moments use denominator n; kurtosis is excess."""
    return pd.DataFrame({
        'mean': data.mean(), 'variance': data.var(ddof=0),
        'skewness': stats.skew(data, bias=True),
        'excess_kurtosis': stats.kurtosis(data, fisher=True, bias=True)}, index=data.columns)

def tails(u):
    """Count marginal 2.5% simultaneous lower and upper tail events."""
    return pd.DataFrame([{'pair': f'{u.columns[i]}-{u.columns[j]}',
                         'lower': int(((u.iloc[:, i] <= .025) & (u.iloc[:, j] <= .025)).sum()),
                         'upper': int(((u.iloc[:, i] >= .975) & (u.iloc[:, j] >= .975)).sum())}
                        for i in range(3) for j in range(i + 1, 3)]).set_index('pair')

def main():
    x = read(1).drop(columns='Day')
    observed = x.notna().astype(int)
    counts = observed.T @ observed
    counts.to_csv(TAB / 'p1_joint_counts.csv')
    print('P1 joint counts\n', counts, '\nComplete rows:', len(x.dropna()))

    r = read(2).Price.pct_change().dropna()
    r -= r.mean()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(1, len(r)+1), r * 100, lw=.8, color='#236f91')
    ax.set(xlabel='Return day', ylabel='Demeaned arithmetic return (%)', title='Problem 2: observed returns before fitting')
    fig.tight_layout(); fig.savefig(FIG / 'p2_returns.png'); plt.close(fig)
    m = moments(r.to_frame('Return')); m.to_csv(TAB / 'p2_moments.csv')
    print('P2 moments\n', m)

    b = read(3)[['A', 'B']]
    m = moments(b); m.to_csv(TAB / 'p3_moments.csv'); print('P3 moments\n', m)
    defaults = b < -.20
    print('P3 losses >20%:', defaults.sum().to_dict(), 'Either:', defaults.any(axis=1).sum(), 'Both:', defaults.all(axis=1).sum())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, pnl, name in zip(axes, [2e6*b.A, 1e6*(b.A+b.B)], ['All in A', 'Half in each']):
        ax.hist(pnl / 1e6, bins=70, color='#236f91')
        ax.set(title=name, xlabel='One-year P&L ($ million)', yscale='log')
    axes[0].set_ylabel('Scenario count (log scale)')
    fig.tight_layout(); fig.savefig(FIG / 'p3_pnl.png'); plt.close(fig)

    x = read(4)
    m = moments(x); m.to_csv(TAB / 'p4_moments.csv'); print('P4 moments\n', m)
    # Each column gets its own sensitivity calculation, dropping its largest
    # absolute deviation from its sample median. This is diagnostic, not cleaning.
    outliers = []
    for col in x:
        idx = (x[col] - x[col].median()).abs().idxmax()
        y = x[[col]].drop(index=idx)
        row = moments(y).iloc[0].to_dict()
        row.update(asset=col, removed_day=int(idx+1), removed_value=float(x.loc[idx, col]))
        outliers.append(row)
    out = pd.DataFrame(outliers).set_index('asset')
    out.to_csv(TAB / 'p4_outlier_sensitivity.csv'); print('P4 leave-one-out\n', out)
    u = x.rank() / (len(x)+1)
    t = tails(u); t.to_csv(TAB / 'p4_observed_tails.csv'); print('P4 rank tail counts\n', t)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
    for ax, (a, b) in zip(axes, [('X1','X2'), ('X1','X3'), ('X2','X3')]):
        ax.scatter(u[a], u[b], s=5, alpha=.45, color='#236f91')
        ax.set(xlabel=a+' rank/(n+1)', ylabel=b+' rank/(n+1)', xlim=(0,1), ylim=(0,1))
        for v in [.025,.975]:
            ax.axhline(v, color='#b54e36', lw=.6); ax.axvline(v, color='#b54e36', lw=.6)
    fig.tight_layout(); fig.savefig(FIG / 'p4_rank_pairs.png'); plt.close(fig)

if __name__ == '__main__':
    main()
