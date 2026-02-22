"""
Monte Carlo Simulation: The True Cost of Razor-and-Blade Pricing
================================================================
Simulates 10,000 five-year ownership scenarios across 5 consumer products,
modeling heterogeneous consumer discount rates (Hausman 1979) and stochastic
usage/pricing patterns.

Author: Fabio Oliveira
For: Towards Data Science article
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import truncnorm, beta
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

np.random.seed(42)
N_SIMULATIONS = 10_000
YEARS = 5
OUTPUT_DIR = Path("/home/claude/charts")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

CATEGORY_COLORS = {
    'Printer': '#2196F3',
    'Coffee': '#795548',
    'Razor': '#4CAF50',
    'Gaming': '#9C27B0',
}

# ─── Product Definitions ─────────────────────────────────────────────────────
# All prices in BRL (R$), sourced from Brazilian retail (Feb 2025)

PRODUCTS = {
    'HP DeskJet 2874': {
        'entry_price': 360,
        'consumable_unit_price': {'mean': 160, 'std': 20},     # HP 667 combo cartridge
        'units_per_year': {'mean': 6, 'std': 2},               # cartridge replacements/yr
        'lock_in_strength': 0.85,                               # DRM + Dynamic Security
        'category': 'Printer',
        'short_name': 'HP DeskJet',
    },
    'Epson EcoTank L3210': {
        'entry_price': 850,
        'consumable_unit_price': {'mean': 56, 'std': 10},      # T544 ink bottle kit
        'units_per_year': {'mean': 1.3, 'std': 0.5},           # refills/yr
        'lock_in_strength': 0.20,                               # open tank system
        'category': 'Printer',
        'short_name': 'Epson EcoTank',
    },
    'Nespresso Essenza Mini': {
        'entry_price': 500,
        'consumable_unit_price': {'mean': 3.40, 'std': 0.30},  # per capsule
        'units_per_year': {'mean': 730, 'std': 180},            # ~2/day +/- variance
        'lock_in_strength': 0.95,                               # Vertuo barcode DRM
        'category': 'Coffee',
        'short_name': 'Nespresso',
    },
    'Gillette Fusion 5': {
        'entry_price': 45,
        'consumable_unit_price': {'mean': 17.50, 'std': 3.0},  # per cartridge
        'units_per_year': {'mean': 52, 'std': 12},              # weekly +/- variance
        'lock_in_strength': 0.30,                               # design complexity only
        'category': 'Razor',
        'short_name': 'Gillette',
    },
    'PS5 Slim Digital': {
        'entry_price': 2900,
        'consumable_unit_price': {'mean': 280, 'std': 60},     # avg game price
        'units_per_year': {'mean': 7, 'std': 3},               # games + PS Plus
        'lock_in_strength': 0.90,                               # ecosystem lock
        'category': 'Gaming',
        'short_name': 'PS5',
    },
}


# ─── Simulation Engine ───────────────────────────────────────────────────────

def truncnorm_rvs(mean, std, low_mult=0.3, high_mult=2.5, size=None):
    """Draw from truncated normal, preventing negative or extreme values."""
    a = (mean * low_mult - mean) / std
    b = (mean * high_mult - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


def simulate_tco(params, n_sims=N_SIMULATIONS, years=YEARS):
    """
    Monte Carlo simulation of Total Cost of Ownership.

    Models heterogeneous consumers with different implicit discount rates
    (Hausman 1979, Beta(2,7) -> mean ~0.22) and stochastic usage/pricing.

    Returns:
        actual_tco:   What consumers really pay over `years`
        perceived_tco: What consumers *think* they'll pay (discounted)
        discount_rates: Each consumer's implicit discount rate
        annual_costs:  (n_sims, years) matrix of annual consumable costs
    """
    entry = params['entry_price']
    price_p = params['consumable_unit_price']
    usage_p = params['units_per_year']

    # Consumer discount rates: Beta(2, 7) ~ mean 0.22, skewed right
    # Calibrated to Hausman (1979): 20-25% implicit discount rates
    discount_rates = beta.rvs(2, 7, size=n_sims)

    # Generate stochastic prices and usage per year
    prices = truncnorm_rvs(price_p['mean'], price_p['std'],
                           low_mult=0.5, high_mult=1.8,
                           size=(n_sims, years))
    usage = truncnorm_rvs(usage_p['mean'], usage_p['std'],
                          low_mult=0.0, high_mult=2.5,
                          size=(n_sims, years))

    annual_costs = prices * usage

    # Actual TCO: entry + sum of all annual costs (no discounting)
    actual_tco = entry + annual_costs.sum(axis=1)

    # Perceived TCO: entry + discounted annual costs
    # This models what a consumer THINKS they'll pay at the point of purchase
    perceived_tco = np.full(n_sims, float(entry))
    for yr in range(years):
        perceived_tco += annual_costs[:, yr] / (1 + discount_rates) ** (yr + 1)

    return actual_tco, perceived_tco, discount_rates, annual_costs


def run_all_simulations():
    """Run Monte Carlo for all products, return structured results dict."""
    results = {}
    for name, params in PRODUCTS.items():
        actual, perceived, rates, annual = simulate_tco(params)
        results[name] = {
            'actual_tco': actual,
            'perceived_tco': perceived,
            'discount_rates': rates,
            'annual_costs': annual,
            'myopia_gap': actual - perceived,
            'params': params,
        }
    return results


# ─── Chart 1: TCO Distribution Grid (2x3) ────────────────────────────────────

def chart_tco_distributions(results):
    """2x3 histogram grid with KDE, median, and 90% CI for each product."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        color = CATEGORY_COLORS[data['params']['category']]
        tco = data['actual_tco']

        ax.hist(tco, bins=60, density=True, alpha=0.65, color=color,
                edgecolor='white', linewidth=0.5)

        # KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(tco)
        x_range = np.linspace(tco.min() * 0.9, tco.max() * 1.1, 300)
        ax.plot(x_range, kde(x_range), color='black', linewidth=1.5, alpha=0.8)

        # Stats
        median = np.median(tco)
        p5, p95 = np.percentile(tco, [5, 95])

        ax.axvline(median, color='#E53935', linestyle='--', linewidth=2,
                   label=f'Median: R${median:,.0f}')
        ax.axvspan(p5, p95, alpha=0.08, color='red',
                   label=f'90% CI: R${p5:,.0f} - R${p95:,.0f}')

        ax.set_title(data['params']['short_name'], fontsize=13, fontweight='bold')
        ax.set_xlabel('5-Year TCO (R$)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8, loc='upper right')

    axes[5].set_visible(False)
    fig.suptitle('Monte Carlo TCO Distributions (n = 10,000 simulations per product)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / '01_tco_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Chart 1: TCO Distributions saved")


# ─── Chart 2: Overlaid KDE Comparison ─────────────────────────────────────────

def chart_overlaid_kde(results):
    """All products on one KDE plot for direct comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    from scipy.stats import gaussian_kde

    for name, data in results.items():
        tco = data['actual_tco']
        color = CATEGORY_COLORS[data['params']['category']]
        kde = gaussian_kde(tco)
        x = np.linspace(0, tco.max() * 1.15, 500)
        ax.fill_between(x, kde(x), alpha=0.25, color=color)
        ax.plot(x, kde(x), linewidth=2, color=color,
                label=f"{data['params']['short_name']} (med: R${np.median(tco):,.0f})")

    ax.set_xlabel('5-Year Total Cost of Ownership (R$)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('TCO Density Comparison Across Product Categories',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / '02_overlaid_kde.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Chart 2: Overlaid KDE saved")


# ─── Chart 3: Box Plot Comparison ─────────────────────────────────────────────

def chart_box_comparison(results):
    """Side-by-side box plots comparing all products."""
    fig, ax = plt.subplots(figsize=(12, 6))

    data_list = []
    for name, data in results.items():
        df_temp = pd.DataFrame({
            'TCO': data['actual_tco'],
            'Product': data['params']['short_name'],
            'Category': data['params']['category'],
        })
        data_list.append(df_temp)
    df = pd.concat(data_list, ignore_index=True)

    order = sorted(results.keys(),
                   key=lambda k: np.median(results[k]['actual_tco']))
    order_short = [results[k]['params']['short_name'] for k in order]
    palette = {results[k]['params']['short_name']: CATEGORY_COLORS[results[k]['params']['category']]
               for k in results}

    sns.boxplot(data=df, x='Product', y='TCO', order=order_short,
                palette=palette, ax=ax, showfliers=False, width=0.6)

    # Add median annotations
    for i, prod in enumerate(order_short):
        med = df[df['Product'] == prod]['TCO'].median()
        ax.text(i, med + 200, f'R${med:,.0f}', ha='center', fontsize=9,
                fontweight='bold', color='#333')

    ax.set_ylabel('5-Year TCO (R$)', fontsize=12)
    ax.set_xlabel('')
    ax.set_title('TCO Distribution Comparison (outliers hidden for clarity)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / '03_box_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Chart 3: Box Plot saved")


# ─── Chart 4: Myopia Tax Bar Chart ────────────────────────────────────────────

def chart_myopia_tax(results):
    """Horizontal bar chart: median gap between actual and perceived TCO."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [results[k]['params']['short_name'] for k in results]
    gaps = [np.median(results[k]['myopia_gap']) for k in results]
    colors = [CATEGORY_COLORS[results[k]['params']['category']] for k in results]

    # Sort by gap
    sorted_idx = np.argsort(gaps)
    names = [names[i] for i in sorted_idx]
    gaps = [gaps[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    bars = ax.barh(names, gaps, color=colors, edgecolor='white', linewidth=1.5,
                   height=0.6)

    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_width() + max(gaps) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'R${gap:,.0f}', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Median "Myopia Gap" (R$)\nActual TCO minus Perceived TCO at Purchase',
                  fontsize=11)
    ax.set_title('The "Myopia Tax": How Much More You Pay Than You Think\n'
                 'Based on Hausman (1979) implicit discount rates, Beta(2,7)',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / '04_myopia_tax.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Chart 4: Myopia Tax saved")


# ─── Chart 5: Tornado Diagram (Sensitivity Analysis) ─────────────────────────

def chart_tornado(results):
    """Tornado diagram: one-at-a-time sensitivity for the 3 worst offenders."""
    targets = ['Nespresso Essenza Mini', 'HP DeskJet 2874', 'Gillette Fusion 5']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, prod_name in zip(axes, targets):
        p = PRODUCTS[prod_name]
        baseline = p['entry_price'] + (p['consumable_unit_price']['mean']
                                       * p['units_per_year']['mean'] * YEARS)

        factors = {}
        # Usage +/- 1 SD
        lo_u = p['entry_price'] + p['consumable_unit_price']['mean'] * (
            p['units_per_year']['mean'] - p['units_per_year']['std']) * YEARS
        hi_u = p['entry_price'] + p['consumable_unit_price']['mean'] * (
            p['units_per_year']['mean'] + p['units_per_year']['std']) * YEARS
        factors['Usage Rate'] = (lo_u - baseline, hi_u - baseline)

        # Price +/- 1 SD
        lo_p = p['entry_price'] + (
            p['consumable_unit_price']['mean'] - p['consumable_unit_price']['std']
        ) * p['units_per_year']['mean'] * YEARS
        hi_p = p['entry_price'] + (
            p['consumable_unit_price']['mean'] + p['consumable_unit_price']['std']
        ) * p['units_per_year']['mean'] * YEARS
        factors['Consumable Price'] = (lo_p - baseline, hi_p - baseline)

        # Discount rate: 10% vs 35% (perceived TCO only)
        pv_lo = p['entry_price'] + sum(
            p['consumable_unit_price']['mean'] * p['units_per_year']['mean']
            / (1.10) ** (y + 1) for y in range(YEARS))
        pv_hi = p['entry_price'] + sum(
            p['consumable_unit_price']['mean'] * p['units_per_year']['mean']
            / (1.35) ** (y + 1) for y in range(YEARS))
        factors['Discount Rate\n(Perceived)'] = (pv_hi - baseline, pv_lo - baseline)

        # Sort by absolute swing
        sorted_factors = sorted(factors.items(),
                                key=lambda x: abs(x[1][1] - x[1][0]),
                                reverse=True)

        y_pos = range(len(sorted_factors))
        for i, (label, (lo, hi)) in enumerate(sorted_factors):
            color = CATEGORY_COLORS[p['category']]
            ax.barh(i, hi, height=0.5, color=color, alpha=0.8, edgecolor='white')
            ax.barh(i, lo, height=0.5, color=color, alpha=0.5, edgecolor='white')

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels([f[0] for f in sorted_factors], fontsize=10)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel('Change from Baseline TCO (R$)')
        ax.set_title(PRODUCTS[prod_name]['short_name'],
                     fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Sensitivity Analysis: Which Variable Drives TCO Most?\n'
                 '(+/- 1 Standard Deviation from Mean)',
                 fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / '05_tornado.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Chart 5: Tornado Diagram saved")


# ─── Chart 6: Convergence Plot ────────────────────────────────────────────────

def chart_convergence(results):
    """Shows how mean TCO stabilizes as simulation count increases."""
    fig, ax = plt.subplots(figsize=(10, 6))

    checkpoints = np.arange(50, N_SIMULATIONS + 1, 50)

    for name, data in results.items():
        color = CATEGORY_COLORS[data['params']['category']]
        running_means = [data['actual_tco'][:n].mean() for n in checkpoints]
        ax.plot(checkpoints, running_means, color=color, linewidth=1.5,
                label=data['params']['short_name'], alpha=0.85)

    ax.set_xlabel('Number of Simulations', fontsize=12)
    ax.set_ylabel('Running Mean TCO (R$)', fontsize=12)
    ax.set_title('Convergence: Mean TCO Stabilizes by ~2,000 Iterations\n'
                 '(10,000 used for robustness)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / '06_convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Chart 6: Convergence Plot saved")


# ─── Chart 7: Lock-In vs TCO Multiplier Scatter ──────────────────────────────

def chart_lockin_vs_multiplier(results):
    """Scatter plot: lock-in strength (x) vs TCO multiplier (y)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, data in results.items():
        lock_in = data['params']['lock_in_strength']
        multiplier = np.median(data['actual_tco']) / data['params']['entry_price']
        color = CATEGORY_COLORS[data['params']['category']]

        ax.scatter(lock_in, multiplier, s=250, c=color,
                   edgecolors='black', linewidth=1.2, zorder=5)
        ax.annotate(data['params']['short_name'],
                    (lock_in, multiplier),
                    textcoords="offset points", xytext=(12, 8),
                    fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))

    # Quadrant labels
    ax.axhline(y=15, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(x=0.55, color='gray', linestyle=':', alpha=0.4)
    ax.text(0.15, 1.5, 'LOW lock-in\nLOW multiplier\n("Honest" products)',
            fontsize=8, color='gray', ha='center', style='italic')
    ax.text(0.85, 1.5, 'HIGH lock-in\nLOW multiplier\n(Ecosystem play)',
            fontsize=8, color='gray', ha='center', style='italic')
    ax.text(0.15, 80, 'LOW lock-in\nHIGH multiplier\n(Disruption target)',
            fontsize=8, color='gray', ha='center', style='italic')
    ax.text(0.85, 80, 'HIGH lock-in\nHIGH multiplier\n(Maximum extraction)',
            fontsize=8, color='gray', ha='center', style='italic')

    ax.set_xlabel('Lock-In Strength\n(0 = open system, 1 = full DRM/ecosystem lock)',
                  fontsize=12)
    ax.set_ylabel('TCO Multiplier\n(Median 5-Year TCO / Entry Price)',
                  fontsize=12)
    ax.set_title('The Lock-In vs. TCO Multiplier Map\n'
                 'Stronger lock-in enables higher consumable extraction '
                 '(but Gillette breaks the pattern)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / '07_lockin_multiplier.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Chart 7: Lock-In vs Multiplier saved")


# ─── Chart 8 (Bonus): Cumulative Probability ─────────────────────────────────

def chart_cumulative_probability(results):
    """CDF: 'X% chance TCO exceeds Y reais' for worst 3 offenders."""
    fig, ax = plt.subplots(figsize=(10, 6))

    targets = ['Nespresso Essenza Mini', 'HP DeskJet 2874', 'PS5 Slim Digital']

    for name in targets:
        data = results[name]
        color = CATEGORY_COLORS[data['params']['category']]
        sorted_tco = np.sort(data['actual_tco'])
        cdf = np.arange(1, len(sorted_tco) + 1) / len(sorted_tco)

        # Plot as "probability of EXCEEDING" (1 - CDF)
        ax.plot(sorted_tco, 1 - cdf, color=color, linewidth=2,
                label=data['params']['short_name'])

        # Mark the 50% and 10% thresholds
        p50 = np.percentile(data['actual_tco'], 50)
        p90 = np.percentile(data['actual_tco'], 90)
        ax.plot(p50, 0.50, 'o', color=color, markersize=8, zorder=5)
        ax.plot(p90, 0.10, 's', color=color, markersize=8, zorder=5)
        ax.annotate(f'R${p90:,.0f}', (p90, 0.10),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=9, color=color, fontweight='bold')

    ax.set_xlabel('5-Year TCO (R$)', fontsize=12)
    ax.set_ylabel('Probability of Exceeding This Cost', fontsize=12)
    ax.set_title('Exceedance Probability: "What Are the Chances I\'ll Pay More Than X?"',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / '08_cumulative_probability.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Chart 8: Cumulative Probability saved")


# ─── Summary Statistics Table ─────────────────────────────────────────────────

def generate_summary_table(results):
    """Create a summary DataFrame with key stats for each product."""
    rows = []
    for name, data in results.items():
        tco = data['actual_tco']
        perceived = data['perceived_tco']
        gap = data['myopia_gap']
        entry = data['params']['entry_price']

        rows.append({
            'Product': data['params']['short_name'],
            'Category': data['params']['category'],
            'Entry Price (R$)': entry,
            'Median TCO (R$)': np.median(tco),
            '5th Pctl TCO': np.percentile(tco, 5),
            '95th Pctl TCO': np.percentile(tco, 95),
            'TCO Multiplier': np.median(tco) / entry,
            'Median Perceived TCO': np.median(perceived),
            'Median Myopia Gap (R$)': np.median(gap),
            'Myopia Gap %': np.median(gap) / np.median(tco) * 100,
            'Lock-In Strength': data['params']['lock_in_strength'],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('Median TCO (R$)', ascending=False)
    return df


# ─── Main Execution ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("  MONTE CARLO SIMULATION: Razor-and-Blade TCO Analysis")
    print(f"  {N_SIMULATIONS:,} simulations x {YEARS} years x {len(PRODUCTS)} products")
    print("=" * 65)
    print()

    # Run simulations
    print("[1/10] Running simulations...")
    results = run_all_simulations()
    print(f"       Done. {N_SIMULATIONS * len(PRODUCTS):,} total scenarios computed.\n")

    # Generate all charts
    print("[2/10] Generating charts...\n")
    chart_tco_distributions(results)
    chart_overlaid_kde(results)
    chart_box_comparison(results)
    chart_myopia_tax(results)
    chart_tornado(results)
    chart_convergence(results)
    chart_lockin_vs_multiplier(results)
    chart_cumulative_probability(results)

    # Summary table
    print("\n[9/10] Generating summary table...\n")
    summary = generate_summary_table(results)

    # Print formatted summary
    print("=" * 90)
    print("  SIMULATION RESULTS SUMMARY")
    print("=" * 90)
    for _, row in summary.iterrows():
        print(f"\n  {row['Product']} ({row['Category']})")
        print(f"    Entry Price:      R${row['Entry Price (R$)']:>8,.0f}")
        print(f"    Median 5yr TCO:   R${row['Median TCO (R$)']:>8,.0f}  "
              f"(90% CI: R${row['5th Pctl TCO']:,.0f} - R${row['95th Pctl TCO']:,.0f})")
        print(f"    TCO Multiplier:   {row['TCO Multiplier']:>8.1f}x")
        print(f"    Myopia Gap:       R${row['Median Myopia Gap (R$)']:>8,.0f}  "
              f"({row['Myopia Gap %']:.1f}% of actual TCO)")
        print(f"    Lock-In Strength: {row['Lock-In Strength']:>8.2f}")
    print()

    # Save summary CSV
    summary.to_csv(OUTPUT_DIR / 'simulation_summary.csv', index=False)
    print(f"[10/10] Summary CSV saved to {OUTPUT_DIR / 'simulation_summary.csv'}")
    print(f"\n  All charts saved to: {OUTPUT_DIR}/")
    print("  Done!")
