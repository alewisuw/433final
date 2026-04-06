# Uplift Modeling Dashboard

This project includes:
- A Streamlit dashboard for uplift simulation (`app.py`)
- Two trained uplift model bundles:
  - `s_learner.pkl`
  - `t_learner.pkl`
- Training notebooks:
  - `s-learner.ipynb`
  - `t-learner.ipynb`

## How to run the dashboard

1. Create/activate your virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Streamlit:

```bash
streamlit run app.py
```

## How to use the dashboard

1. Open the Streamlit app in your browser.
2. In the sidebar:
- Upload a CSV, or use the default `test.csv`.
- Choose a model (`T-learner` or `S-learner`).
- Set business assumptions:
  - Price per unit
  - Variable cost per unit
  - Marketing cost per treated user
- Set `% of people treated`.
- Optionally click **Find optimal split** to optimize for `Profit` or `Revenue`.
3. Review outputs:
- Scenario comparison table (`Treat none`, `Treat all`, `Random split`, `Optimized uplift`)
- Conversion/revenue/profit comparisons
- Sensitivity charts vs treatment percentage
4. Download scored predictions as CSV.

## Model summary

### T-learner
- Uses two separate models:
  - one trained on treated users
  - one trained on control users
- Uplift is computed as:
  - `P(conversion | treated) - P(conversion | control)`

### S-learner
- Uses one model with treatment as a feature.
- The app scores each row twice:
  - once with treatment set to `1`
  - once with treatment set to `0`
- Uplift is computed as the difference between those two predicted probabilities.

## About `train.csv`

`train.csv` is not included in this GitHub repository because the file is too large for GitHub file size limits.

The training data is an 80/20 split derived from this dataset:
- https://www.kaggle.com/datasets/arashnic/uplift-modeling

Use the Kaggle dataset above to recreate `train.csv` locally if needed.
