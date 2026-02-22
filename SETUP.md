# 🚀 GitHub Setup Instructions

## Step 1: Create the Repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `razor-blade-monte-carlo`
3. Description: "Monte Carlo simulation of razor-and-blade pricing TCO across 5 consumer products, with behavioral economics framework (Gabaix & Laibson, Hausman, Williamson)"
4. Set to **Public**
5. **Do NOT** initialize with README (we already have one)
6. Click "Create repository"

## Step 2: Push This Repo

```bash
cd razor-blade-monte-carlo
git init
git add .
git commit -m "feat: initial commit — 10,000-iteration Monte Carlo TCO simulation

- Simulate 5 razor-and-blade products with stochastic parameters
- Model heterogeneous consumer discount rates (Hausman 1979, Beta(2,7))
- Generate 8 publication-quality charts (150 DPI, matplotlib/seaborn)
- Quantify 'Myopia Tax' across printers, coffee, razors, gaming consoles
- Brazilian market data (R$, Feb 2025)
- Companion article for Towards Data Science"

git branch -M main
git remote add origin https://github.com/fabionoliveirastr/razor-blade-monte-carlo.git
git push -u origin main
```

## Step 3: Add Topics (for discoverability)

Go to repository Settings → Topics, add:
- `monte-carlo-simulation`
- `behavioral-economics`
- `pricing-strategy`
- `data-science`
- `python`
- `total-cost-of-ownership`
- `consumer-behavior`

## Step 4: Enable GitHub Pages (optional, for charts)

Settings → Pages → Source: Deploy from branch → Branch: main, /docs → Save

## Step 5: Pin the Repository

Go to your profile → "Customize your pins" → Select this repo

---

**After pushing, replace all `fabionoliveirastr` and `YOUR_LINKEDIN` placeholders in README.md and the notebook.**
