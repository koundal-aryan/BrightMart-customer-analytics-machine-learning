# BrightMart Retail — Customer Analytics & Machine Learning
## Data-Driven Customer Segmentation, Sales Prediction & Marketing Insights

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://brightmart-retail-analytics.streamlit.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/koundal-aryan/BrightMart-customer-analytics-machine-learning)
[![Academic Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://github.com/koundal-aryan/BrightMart-customer-analytics-machine-learning/blob/main/customer_analytics_ml_project.ipynb)

An end-to-end analytical modelling and machine learning project built for customer analytics, sales forecasting, customer segmentation, and marketing campaign propensity prediction.

---

## 🌐 Live Streamlit Dashboard & Project Links

- 🚀 **Live Streamlit Dashboard:** [https://brightmart-retail-analytics.streamlit.app](https://brightmart-retail-analytics.streamlit.app) *(or deploy via [Streamlit Cloud](https://share.streamlit.io))*
- 📓 **Interactive Jupyter Notebook:** [View Notebook with Rendered Charts](https://github.com/koundal-aryan/BrightMart-customer-analytics-machine-learning/blob/main/customer_analytics_ml_project.ipynb)
- 📦 **GitHub Repository:** [https://github.com/koundal-aryan/BrightMart-customer-analytics-machine-learning](https://github.com/koundal-aryan/BrightMart-customer-analytics-machine-learning)

---

## Project Repository & Implementation Structure
This project is dedicated strictly to Customer Analytics & Retail Machine Learning. It contains ONLY the required files:

```
BrightMart-customer-analytics-machine-learning/
│
├── BrightMart_ML_Dataset.csv                  # The actual 648-record retail dataset
├── customer_analytics_ml_project.ipynb        # Complete 16-section academic Jupyter Notebook (with rendered plots)
├── app.py                                     # Professional interactive Streamlit dashboard
├── requirements.txt                           # Minimal deployment dependencies
├── .gitignore                                 # Git configuration
└── README.md                                  # Project overview and deployment guide
```

---

## 1. How to Run Locally

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Run the Jupyter Notebook
```bash
jupyter notebook customer_analytics_ml_project.ipynb
```

---

## 2. Streamlit Community Cloud Deployment

To deploy your live app on Streamlit Cloud:

1. Open **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"**
3. Select:
   - **Repository:** `koundal-aryan/BrightMart-customer-analytics-machine-learning`
   - **Branch:** `main` (or `master`)
   - **Main file path:** `app.py`
   - **App URL:** `brightmart-retail-analytics.streamlit.app`
4. Click **"Deploy"**
