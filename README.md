# 🔬 The Hidden Math Behind "Cheap" Printers

**Monte Carlo Simulation Reveals the True Cost of Razor-and-Blade Pricing**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Towards Data Science](https://img.shields.io/badge/Published%20on-TDS-blue)](https://towardsdatascience.com/)

> Using Python to simulate 10,000 five-year ownership scenarios across printers, coffee capsules, razors, and gaming consoles — and what behavioral economics says about why we keep falling for it.

---

## 📊 Key Findings

| Product | Entry Price | Median 5yr TCO | Multiplier | "Myopia Tax" |
|---------|-----------|---------------|------------|-------------|
| Nespresso Essenza Mini | R$500 | R$12,898 | **25.8×** | R$4,921 (38%) |
| PS5 Slim Digital | R$2,900 | R$12,816 | 4.4× | R$3,792 (30%) |
| HP DeskJet 2874 | R$360 | R$5,155 | **14.3×** | R$1,890 (37%) |
| Gillette Fusion 5 | R$45 | R$4,575 | **101.7×** | R$1,808 (40%) |
| Epson EcoTank L3210 | R$850 | R$1,214 | 1.4× | R$142 (12%) |

The **"Myopia Tax"** measures how much consumers underestimate their 5-year spending due to implicit discounting of future costs ([Hausman, 1979](https://www.jstor.org/stable/3003415)).

## 🧠 Theoretical Framework

This project sits at the intersection of **data science** and **behavioral economics**, building on three foundational theories:

- **Shrouded Attributes** — [Gabaix & Laibson (2006, *QJE*)](https://doi.org/10.1162/qjec.2006.121.2.505): Market competition does not eliminate cost-hiding when consumers are myopic
- **Implicit Discount Rates** — [Hausman (1979)](https://www.jstor.org/stable/3003415): Consumers apply ~20-25% discount rates to future costs
- **Hold-Up Problem** — [Williamson (1985)](https://en.wikipedia.org/wiki/Oliver_E._Williamson): Relationship-specific investment (buying the hardware) destroys future bargaining power

## 📈 Visualizations

The notebook generates 8 publication-quality charts:

| Chart | Description |
|-------|-------------|
| TCO Distributions | 2×3 histogram grid with KDE, median, and 90% CI |
| Overlaid KDE | All products on one density plot |
| Box Plot | Side-by-side distribution comparison |
| Myopia Tax | Horizontal bar chart of median perception gaps |
| Tornado Diagram | Sensitivity analysis for 3 worst offenders |
| Convergence Plot | Mean TCO stabilization vs. iteration count |
| Lock-In Map | Scatter: lock-in strength vs. TCO multiplier |
| Exceedance Probability | CDF: "X% chance TCO exceeds Y reais" |

## 🛠 Simulation Architecture

**10,000 iterations** per product using stochastic sampling:

| Variable | Distribution | Rationale |
|----------|-------------|-----------|
| Usage rate | Truncated Normal | Bounded ≥0; captures habitual consumption variance |
| Consumable price | Truncated Normal | Market price fluctuation within observed range |
| Consumer discount rate | Beta(2, 7) | Right-skewed, mean ≈ 0.22; calibrated to Hausman (1979) |

**TCO Formula per consumer *i*:**

```
Actual TCO_i    = Entry Price + Σ(Usage_t × Price_t) for t=1..5
Perceived TCO_i = Entry Price + Σ(Usage_t × Price_t) / (1 + r_i)^t for t=1..5
Myopia Gap_i    = Actual TCO_i - Perceived TCO_i
```

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.9
```

### Installation

```bash
git clone https://github.com/fabionoliveirastr/razor-blade-monte-carlo.git
cd razor-blade-monte-carlo
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook notebooks/razor_blade_monte_carlo.ipynb
```

Or run the standalone script:

```bash
python src/razor_blade_monte_carlo.py
```

### Customize for Your Products

Edit the `PRODUCTS` dictionary in the notebook to simulate any razor-and-blade product with your local market prices:

```python
PRODUCTS['Your Product'] = {
    'entry_price': 100,                                    # base product price
    'consumable_unit_price': {'mean': 10, 'std': 2},       # per-unit consumable
    'units_per_year': {'mean': 50, 'std': 10},             # annual consumption
    'lock_in_strength': 0.70,                               # 0-1 scale
    'category': 'YourCategory',
    'short_name': 'Short Name',
}
```

## 📁 Repository Structure

```
razor-blade-monte-carlo/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── razor_blade_monte_carlo.ipynb    # Main notebook (start here)
├── src/
│   └── razor_blade_monte_carlo.py       # Standalone simulation script
├── charts/                              # Generated visualizations (150 DPI)
│   ├── 01_tco_distributions.png
│   ├── 02_overlaid_kde.png
│   ├── 03_box_comparison.png
│   ├── 04_myopia_tax.png
│   ├── 05_tornado.png
│   ├── 06_convergence.png
│   ├── 07_lockin_multiplier.png
│   └── 08_cumulative_probability.png
├── data/
│   └── simulation_summary.csv           # Summary statistics
└── docs/
    └── article_draft.md                 # TDS article draft
```

## 📚 References

1. Gabaix, X. & Laibson, D. (2006). "Shrouded Attributes, Consumer Myopia, and Information Suppression in Competitive Markets." *Quarterly Journal of Economics*, 121(2).
2. Hausman, J. (1979). "Individual Discount Rates and the Purchase and Utilization of Energy-Using Durables." *Bell Journal of Economics*, 10(1).
3. Allcott, H. & Wozny, N. (2014). "Gasoline Prices, Fuel Economy, and the Energy Paradox." *Review of Economics and Statistics*, 96(5).
4. Williamson, O. (1985). *The Economic Institutions of Capitalism*. Free Press.
5. Morwitz, V., Greenleaf, E. & Johnson, E. (1998). "Divide and Prosper: Consumers' Reactions to Partitioned Prices." *Journal of Marketing Research*, 35(4).

## 📝 Companion Article

This repository accompanies an article published on [Towards Data Science](https://towardsdatascience.com/). The article provides a narrative walkthrough of the methodology, results, and implications.

## 👤 Author

**Fabio Oliveira**  
Electronic & Computer Engineer (UFRJ — 10.0/10.0 thesis) | Strategy & Operations (ex-QuintoAndar, ex-OLX Brasil) | Harvard Business School CORe

- [LinkedIn](https://www.linkedin.com/in/fabionoliveirastr/)
- [Towards Data Science](https://towardsdatascience.com/)

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
