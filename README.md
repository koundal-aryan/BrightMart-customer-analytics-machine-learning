# BrightMart Retail — Customer Analytics & Machine Learning

> **Mini Project | Analytical Modelling with Machine Learning**

---

## 📊 Project Overview

A complete machine learning analytics project for BrightMart Retail, covering data preparation, EDA, regression, classification, and clustering on a synthetic retail customer dataset.

| Item | Detail |
|------|--------|
| **Dataset** | `BrightMart_ML_Dataset.csv` — 648 customer records, 10 features |
| **Notebook** | `customer_analytics_ml_project.ipynb` — all tasks A–F with visible code & plots |
| **Dashboard** | `dashboard.html` — self-contained HTML dashboard (open in any browser) |
| **Author** | Aryan Koundal |

---

## 🧠 ML Models & Results

| Model | Task | Key Metric |
|-------|------|-----------|
| **Linear Regression** | Sales Prediction | R² = 0.8727, MAE = ₹13.10k |
| **Decision Tree** | Campaign Response | Accuracy = 77.69%, F1 = 0.8199 |
| **K-Means (K=3)** | Customer Segmentation | 3 clusters: Premium Champions, Budget Conscious, Aspirational Enthusiasts |
| **Random Forest** | Campaign Response | Accuracy = 88.46%, F1 = 0.9123 |

---

## 📁 Files

```
BrightMart-customer-analytics-machine-learning/
├── customer_analytics_ml_project.ipynb   ← Main notebook (Tasks A–F + Q&A)
├── dashboard.html                         ← Standalone HTML dashboard (no server needed)
├── BrightMart_ML_Dataset.csv             ← Cleaned dataset (648 records)
├── requirements.txt                       ← Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Run the Notebook
```bash
pip install -r requirements.txt
jupyter notebook customer_analytics_ml_project.ipynb
```

### View the Dashboard
Simply open `dashboard.html` in any browser — no server required:
```bash
open dashboard.html        # macOS
start dashboard.html       # Windows
xdg-open dashboard.html    # Linux
```

---

## 📈 Tasks Completed

| Task | Description | Status |
|------|-------------|--------|
| **Task A** | Data Preparation & EDA (2+ visualizations) | ✅ |
| **Task B** | Linear Regression — Sales Prediction | ✅ |
| **Task C** | Decision Tree Classifier | ✅ |
| **Task D** | K-Means Customer Segmentation | ✅ |
| **Task E** | Random Forest + Model Comparison | ✅ |
| **Task F** | Business Recommendations & Limitations | ✅ |
| **Q1–Q9** | 9 Analysis Questions Answered | ✅ |

---

## 🔗 Links

- **GitHub Repository:** [koundal-aryan/BrightMart-customer-analytics-machine-learning](https://github.com/koundal-aryan/BrightMart-customer-analytics-machine-learning)
- **Dashboard:** Open `dashboard.html` locally in your browser
- **Notebook:** `customer_analytics_ml_project.ipynb` (view on GitHub or run locally)
