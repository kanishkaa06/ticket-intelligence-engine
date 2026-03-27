# Support Ticket Intelligence Engine

An end-to-end ML-powered support operations platform that goes beyond
basic dashboards — classifying tickets, predicting SLA breaches before
they happen, clustering root causes from raw text, and detecting hidden
agent overload by complexity rather than volume.

**Live Demo:** [Click to open app](#) ← we will update this link after deployment

---

## The Problem This Solves

Most support teams track three things: ticket volume, SLA percentage,
and category pie charts. None of these prevent problems — they just
describe them after the fact.

This system predicts and explains:
- What priority should this ticket be? (before a human decides)
- Will this ticket breach its SLA? (before the deadline passes)
- What hidden issue patterns exist across thousands of tickets? (without manual tagging)
- Which agents are overloaded by complexity — not just count?

---

## Modules

| Module | Technique | Result |
|---|---|---|
| Ticket Triage Classifier | Random Forest | 91% accuracy, 0.94 F1 on Urgent |
| SLA Breach Predictor | Random Forest | 0.717 AUC-ROC |
| Root Cause Clustering | TF-IDF + K-Means | 6 clean issue clusters |
| Workforce Load Analyzer | Complexity-weighted scoring | 7x load disparity detected |

---

## Key Finding — Feature Importance

Resolution time and response time together account for **66% of
predictive signal** for ticket priority. Category and channel contribute
less than 3% each — meaning operational behaviour, not ticket metadata,
drives priority classification.

---

## Data Engineering Decision

Two publicly available Kaggle datasets were evaluated and rejected after
EDA revealed random label assignment — SLA breach rates were statistically
identical across all priority levels (50% each), indicating synthetic
label noise. A domain-realistic dataset was engineered from scratch using
industry-standard SLA thresholds:

| Priority | Response SLA | Resolution SLA | Target Breach Rate |
|---|---|---|---|
| Urgent | 1 hour | 8 hours | 20% |
| High | 4 hours | 24 hours | 40% |
| Medium | 12 hours | 48 hours | 60% |
| Low | 24 hours | 120 hours | 75% |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & EDA | Python, Pandas, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost, Random Forest |
| NLP | TF-IDF Vectorizer, K-Means Clustering, PCA |
| Dashboard | Streamlit |
| Environment | Jupyter Notebook, VS Code |

---

## How to Run Locally
```bash
# Clone the repo
git clone https://github.com/kanishkaa06/ticket-intelligence-engine.git
cd ticket-intelligence-engine

# Install dependencies
pip install pandas scikit-learn matplotlib seaborn streamlit shap

# Run the dashboard
streamlit run dashboard.py
```

---

## Project Structure
```
ticket-intelligence-engine/
│
├── dashboard.py              # Streamlit app — all 5 pages
├── 01_exploration.ipynb      # Full EDA and model development notebook
├── tickets_clean.csv         # Domain-realistic synthetic dataset
├── customer_support_tickets_200k.csv  # Real NLP source data
└── README.md
```

---

## What I Learned

- How to identify and handle label noise in public datasets
- How to engineer domain-realistic synthetic data when real data
  is unavailable or unreliable
- How TF-IDF converts raw text into ML-readable numbers
- Why complexity-weighted metrics reveal what volume metrics hide
- How to build and deploy an end-to-end ML application

---

*Built by Kanishka Dubey — Computer Science graduate targeting
Data/ML/AI roles*
```

After saving, run these three commands to push the README to GitHub:
```
git add README.md
git commit -m "Add project README"
git push