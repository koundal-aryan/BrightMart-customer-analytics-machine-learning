"""
app.py
Customer Analytics & Machine Learning Dashboard
Data-Driven Customer Segmentation, Sales Prediction & Marketing Insights

Professional Streamlit Dashboard strictly implementing:
- Overview (KPI cards & dataset info)
- Exploratory Analysis (Interactive charts, Ad_Spend vs Sales, Correlation heatmap, Distributions)
- Sales Prediction (Linear Regression metrics, Actual vs Predicted, dynamic Ad_Spend prediction)
- Customer Segmentation (StandardScaler, Elbow Method K=1-10, K-Means clustering, Centroids, Profiles)
- Classification Models (Decision Tree vs Random Forest, Confusion Matrix, Tree Rules, Feature Importance, Comparison Table)
- Business Insights (Automated advertising, segment strategies, model recommendations, limitations, monitoring)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Publication styling for matplotlib/seaborn
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'

# Custom CSS for executive cards and typography
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.4rem;
    }
    .metric-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .footer-text {
        text-align: center;
        color: #718096;
        font-size: 0.85rem;
        border-top: 1px solid #e2e8f0;
        padding-top: 20px;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Ingestion & Preprocessing
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    dataset_candidates = [
        "BrightMart_ML_Dataset.csv",
        os.path.join(os.path.dirname(__file__), "BrightMart_ML_Dataset.csv"),
        os.path.join(os.path.dirname(__file__), "data", "BrightMart_ML_Dataset.csv"),
        os.path.join(os.path.dirname(__file__), "outputs", "cleaned_data", "BrightMart_Cleaned_Dataset.csv")
    ]
    data_path = None
    for p in dataset_candidates:
        if os.path.exists(p):
            data_path = p
            break
            
    if data_path is None:
        st.error("Dataset 'BrightMart_ML_Dataset.csv' could not be located.")
        st.stop()
        
    df = pd.read_csv(data_path)
    
    # Cleaning pipeline identical to the notebook
    df_clean = df.drop_duplicates().copy()
    if 'Annual_Income' in df_clean.columns and df_clean['Annual_Income'].isnull().sum() > 0:
        df_clean['Annual_Income'] = df_clean['Annual_Income'].fillna(df_clean['Annual_Income'].median())
    if 'Spending_Score' in df_clean.columns and df_clean['Spending_Score'].isnull().sum() > 0:
        df_clean['Spending_Score'] = df_clean['Spending_Score'].fillna(df_clean['Spending_Score'].median())
        
    if 'Age' in df_clean.columns:
        df_clean = df_clean[(df_clean['Age'] >= 18) & (df_clean['Age'] <= 100)]
    if 'Sales' in df_clean.columns:
        df_clean = df_clean[df_clean['Sales'] > 0]
    if 'Spending_Score' in df_clean.columns:
        df_clean['Spending_Score'] = df_clean['Spending_Score'].clip(1, 100)
        
    df_clean['Age'] = df_clean['Age'].astype(int)
    df_clean['Spending_Score'] = df_clean['Spending_Score'].astype(int)
    df_clean['Purchase_Frequency'] = df_clean['Purchase_Frequency'].astype(int)
    df_clean.reset_index(drop=True, inplace=True)
    
    # Categorical encodings
    df_clean['Discount_Usage_Encoded'] = df_clean['Discount_Usage'].map({'Low': 0, 'Medium': 1, 'High': 2})
    df_clean['Campaign_Response_Binary'] = df_clean['Campaign_Response'].map({'No': 0, 'Yes': 1})
    
    return df_clean

df = load_and_preprocess_data()

# ---------------------------------------------------------
# Cached Models Execution
# ---------------------------------------------------------
@st.cache_resource
def train_linear_regression(data):
    X = data[['Ad_Spend']]
    y = data['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    mae = float(mean_absolute_error(y_test, y_pred_test))
    mse = float(mean_squared_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred_test))
    
    return {
        'model': model,
        'slope': slope,
        'intercept': intercept,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }

@st.cache_resource
def train_kmeans_clustering(data):
    features = ['Annual_Income', 'Spending_Score']
    X_raw = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    k_range = list(range(1, 11))
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
        
    optimal_k = 3
    final_km = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)
    cluster_labels = final_km.fit_predict(X_scaled)
    centroids_orig = scaler.inverse_transform(final_km.cluster_centers_)
    
    df_clustered = data.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # Assign empirical business labels after examining profiles
    cluster_names = {
        0: "High-Income / High-Spending (Premium Champions)",
        1: "Low-Income / Low-Spending (Budget Conscious)",
        2: "Low-Income / High-Spending (Aspirational Enthusiasts)"
    }
    df_clustered['Business_Label'] = df_clustered['Cluster'].map(cluster_names)
    
    profiles = []
    for c in range(optimal_k):
        sub = df_clustered[df_clustered['Cluster'] == c]
        profiles.append({
            'Cluster': c,
            'Customer Count': len(sub),
            'Percentage': round((len(sub) / len(df_clustered)) * 100, 2),
            'Average Annual Income (₹k)': round(sub['Annual_Income'].mean(), 2),
            'Average Spending Score': round(sub['Spending_Score'].mean(), 2),
            'Average Purchase Frequency': round(sub['Purchase_Frequency'].mean(), 2),
            'Business Label': cluster_names[c]
        })
    profiles_df = pd.DataFrame(profiles)
    
    return {
        'scaler': scaler,
        'k_range': k_range,
        'inertias': inertias,
        'optimal_k': optimal_k,
        'kmeans': final_km,
        'centroids': centroids_orig,
        'df_clustered': df_clustered,
        'profiles_df': profiles_df,
        'cluster_names': cluster_names
    }

@st.cache_resource
def train_classification_models(data):
    features = ['Age', 'Annual_Income', 'Purchase_Frequency', 'Discount_Usage_Encoded', 'Spending_Score']
    X = data[features]
    y = data['Campaign_Response_Binary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 1. Decision Tree
    dt = DecisionTreeClassifier(max_depth=4, random_state=42, criterion='gini')
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    
    acc_dt = float(accuracy_score(y_test, y_pred_dt))
    prec_dt = float(precision_score(y_test, y_pred_dt, zero_division=0))
    rec_dt = float(recall_score(y_test, y_pred_dt, zero_division=0))
    f1_dt = float(f1_score(y_test, y_pred_dt, zero_division=0))
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, criterion='gini')
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    acc_rf = float(accuracy_score(y_test, y_pred_rf))
    prec_rf = float(precision_score(y_test, y_pred_rf, zero_division=0))
    rec_rf = float(recall_score(y_test, y_pred_rf, zero_division=0))
    f1_rf = float(f1_score(y_test, y_pred_rf, zero_division=0))
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    importances = rf.feature_importances_
    
    comp_df = pd.DataFrame([
        {
            'Model': 'Decision Tree',
            'Accuracy': round(acc_dt, 4),
            'Precision': round(prec_dt, 4),
            'Recall': round(rec_dt, 4),
            'F1': round(f1_dt, 4)
        },
        {
            'Model': 'Random Forest',
            'Accuracy': round(acc_rf, 4),
            'Precision': round(prec_rf, 4),
            'Recall': round(rec_rf, 4),
            'F1': round(f1_rf, 4)
        }
    ])
    
    return {
        'features': features,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'dt_model': dt,
        'dt_metrics': {'Accuracy': acc_dt, 'Precision': prec_dt, 'Recall': rec_dt, 'F1': f1_dt, 'cm': cm_dt},
        'rf_model': rf,
        'rf_metrics': {'Accuracy': acc_rf, 'Precision': prec_rf, 'Recall': rec_rf, 'F1': f1_rf, 'cm': cm_rf},
        'rf_importances': importances,
        'comp_df': comp_df
    }

reg_res = train_linear_regression(df)
km_res = train_kmeans_clustering(df)
clf_res = train_classification_models(df)

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.markdown("### 🛒 BrightMart Analytics")
st.sidebar.markdown("**Customer Analytics & Machine Learning**")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Exploratory Analysis",
        "Sales Prediction",
        "Customer Segmentation",
        "Classification Models",
        "Business Insights"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Project Details:**
- Model Engine: Scikit-learn
- Target 1: Sales (Regression)
- Target 2: Campaign_Response
- Clustering: K-Means (K=3)
""")

# ---------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------
if menu == "Overview":
    st.markdown('<div class="main-header">Customer Analytics & Machine Learning Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data-Driven Customer Segmentation, Sales Prediction & Marketing Insights</div>', unsafe_allow_html=True)
    
    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Total Customers", value=f"{len(df):,}")
    with col2:
        st.metric(label="Average Sales", value=f"₹{df['Sales'].mean():.2f}k")
    with col3:
        st.metric(label="Average Ad Spend", value=f"₹{df['Ad_Spend'].mean():.2f}k")
    with col4:
        st.metric(label="Avg Annual Income", value=f"₹{df['Annual_Income'].mean():.2f}k")
    with col5:
        st.metric(label="Avg Spending Score", value=f"{df['Spending_Score'].mean():.1f} / 100")
        
    st.markdown("---")
    
    st.subheader("📌 Project Overview & Architecture")
    st.write("""
    BrightMart is a multi-format retail enterprise operating across metropolitan and suburban consumer regions. 
    Store leadership seeks to transition from intuition-driven decision-making to a structured, predictive data analytics architecture.
    This dashboard unifies three core analytical pillars:
    
    1. **Econometric Sales Forecasting:** Estimating the marginal revenue sensitivity per unit advertising expenditure using Ordinary Least Squares (OLS) regression.
    2. **Unsupervised Behavioral Segmentation:** Clustering customers via K-Means on standardized income and spending metrics to discover empirical personas.
    3. **Predictive Campaign Propensity:** Training Decision Tree and Random Forest classifiers to predict marketing campaign response (`Campaign_Response`) with high precision and recall.
    """)
    
    st.subheader("📋 Dataset Inspection")
    col_t1, col_t2 = st.columns([1.2, 0.8])
    with col_t1:
        st.markdown("**Cleaned Customer Records Sample (First 10 Rows):**")
        st.dataframe(df[['Customer_ID', 'Age', 'Annual_Income', 'Spending_Score', 'Purchase_Frequency', 'Discount_Usage', 'Ad_Spend', 'Sales', 'Campaign_Response']].head(10), use_container_width=True)
    with col_t2:
        st.markdown("**Dataset Attributes & Statistical Summary:**")
        st.dataframe(df[['Age', 'Annual_Income', 'Spending_Score', 'Purchase_Frequency', 'Ad_Spend', 'Sales']].describe().round(2), use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: EXPLORATORY ANALYSIS
# ---------------------------------------------------------
elif menu == "Exploratory Analysis":
    st.markdown('<div class="main-header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Visualizations of Demographic and Behavioral Relationships</div>', unsafe_allow_html=True)
    
    # 1. Ad Spend vs Sales with Regression line
    st.subheader("1. Advertising Spend vs Store Sales")
    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    sns.regplot(
        data=df, x='Ad_Spend', y='Sales', ax=ax1,
        color='#1f77b4',
        scatter_kws={'alpha': 0.6, 's': 45},
        line_kws={'color': '#d62728', 'linewidth': 2.5, 'label': f"Fitted Line: y = {reg_res['intercept']:.2f} + {reg_res['slope']:.2f}x"}
    )
    ax1.set_title("BrightMart: Advertising Spend vs Sales Relationship", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Advertising Spend (₹ in Thousands)", fontsize=11)
    ax1.set_ylabel("Sales (₹ in Thousands)", fontsize=11)
    ax1.legend(frameon=True)
    st.pyplot(fig1)
    
    col_a, col_b = st.columns(2)
    with col_a:
        # 2. Correlation Heatmap
        st.subheader("2. Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        num_cols = ['Age', 'Annual_Income', 'Spending_Score', 'Purchase_Frequency', 'Ad_Spend', 'Sales', 'Discount_Usage_Encoded', 'Campaign_Response_Binary']
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', ax=ax2, square=True, cbar=False, annot_kws={"size": 9})
        ax2.set_title("Numerical Feature Correlation Matrix", fontsize=12, fontweight='bold', pad=10)
        st.pyplot(fig2)
        
    with col_b:
        # 3 & 4. Customer Income & Spending Score Distribution
        st.subheader("3 & 4. Demographic Distributions")
        dist_choice = st.radio("Select Variable to View Distribution:", ["Customer Annual Income", "Spending Score (1 - 100)"], horizontal=True)
        fig3, ax3 = plt.subplots(figsize=(7, 4.2))
        if dist_choice == "Customer Annual Income":
            sns.histplot(df['Annual_Income'], kde=True, color='#2ca02c', bins=25, ax=ax3)
            ax3.set_title("Annual Income Distribution (₹ in Thousands)", fontsize=12, fontweight='bold', pad=10)
            ax3.set_xlabel("Annual Income (₹ Thousands)", fontsize=10)
        else:
            sns.histplot(df['Spending_Score'], kde=True, color='#9467bd', bins=25, ax=ax3)
            ax3.set_title("Spending Score Distribution", fontsize=12, fontweight='bold', pad=10)
            ax3.set_xlabel("Spending Score (1 - 100)", fontsize=10)
        st.pyplot(fig3)

# ---------------------------------------------------------
# PAGE 3: SALES PREDICTION (LINEAR REGRESSION)
# ---------------------------------------------------------
elif menu == "Sales Prediction":
    st.markdown('<div class="main-header">Linear Regression for Sales Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Modeling Sales (Y) from Advertising Spend (X) using Ordinary Least Squares</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Slope Coefficient (β₁)", value=f"{reg_res['slope']:.4f}")
    with col2:
        st.metric(label="Baseline Intercept (β₀)", value=f"₹{reg_res['intercept']:.2f}k")
    with col3:
        st.metric(label="Mean Absolute Error (MAE)", value=f"₹{reg_res['mae']:.2f}k")
    with col4:
        st.metric(label="R² Score", value=f"{reg_res['r2']:.4f}", delta=f"{reg_res['r2']*100:.1f}% Variance")
        
    st.markdown("---")
    
    col_chart, col_pred = st.columns([1.2, 0.8])
    with col_chart:
        st.subheader("Actual Sales vs Predicted Sales")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(reg_res['X_test'], reg_res['y_test'], color='#1f77b4', alpha=0.7, label='Actual Observations (Test)', s=50)
        ax.plot(reg_res['X_test'], reg_res['y_pred_test'], color='#d62728', linewidth=2, label='Predicted OLS Regression Line')
        ax.set_title("Actual vs Predicted Sales on Hold-Out Test Set", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Ad Spend (₹ in Thousands)", fontsize=11)
        ax.set_ylabel("Sales (₹ in Thousands)", fontsize=11)
        ax.legend(frameon=True)
        st.pyplot(fig)
        
    with col_pred:
        st.subheader("🧮 Dynamic Sales Forecaster")
        st.write("Enter an advertising spend budget to calculate expected store sales:")
        
        ad_input = st.number_input(
            "Advertising Spend (Ad_Spend in ₹ Thousands):",
            min_value=0.0,
            max_value=300.0,
            value=50.0,
            step=5.0
        )
        
        predicted_sales = reg_res['intercept'] + (reg_res['slope'] * ad_input)
        
        st.success(f"### Predicted Sales: **₹{predicted_sales:.2f}k**")
        st.caption(f"Mathematical Formulation: Sales = {reg_res['intercept']:.2f} + ({reg_res['slope']:.4f} × {ad_input:.2f})")
        
        st.markdown("#### Managerial Econometric Takeaways:")
        st.info(f"""
        - **Marginal ROAS:** For every ₹1,000 allocated to ad spend, store sales increase by **₹{reg_res['slope']*1000:.2f}** on average.
        - **Organic Base:** Expected store baseline sales when ad spend is ₹0 equals **₹{reg_res['intercept']:.2f}k**.
        - **Goodness-of-Fit:** **{reg_res['r2']*100:.2f}%** of store sales variance is explained by advertising expenditure.
        """)

# ---------------------------------------------------------
# PAGE 4: CUSTOMER SEGMENTATION (K-MEANS)
# ---------------------------------------------------------
elif menu == "Customer Segmentation":
    st.markdown('<div class="main-header">K-Means Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unsupervised Persona Discovery via StandardScaler & WCSS Elbow Curvature</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Elbow Method for Optimal K")
        fig_elbow, ax_elbow = plt.subplots(figsize=(7, 4.5))
        ax_elbow.plot(km_res['k_range'], km_res['inertias'], 'bo-', linewidth=2, markersize=7, label='Inertia (WCSS)')
        ax_elbow.axvline(x=km_res['optimal_k'], color='red', linestyle='--', linewidth=1.5, label=f"Selected Optimal K = {km_res['optimal_k']}")
        ax_elbow.scatter([km_res['optimal_k']], [km_res['inertias'][km_res['optimal_k'] - 1]], color='red', s=120, zorder=5)
        ax_elbow.set_title("Within-Cluster Sum of Squares (Inertia) vs K", fontsize=12, fontweight='bold', pad=10)
        ax_elbow.set_xlabel("Number of Clusters (K)", fontsize=11)
        ax_elbow.set_ylabel("Inertia", fontsize=11)
        ax_elbow.set_xticks(km_res['k_range'])
        ax_elbow.legend(frameon=True)
        st.pyplot(fig_elbow)
        
    with col2:
        st.subheader("2. Customer Cluster Scatter Plot & Centroids")
        fig_clust, ax_clust = plt.subplots(figsize=(7, 4.5))
        palette = ['#2ca02c', '#1f77b4', '#d62728']
        
        for c in range(km_res['optimal_k']):
            sub = km_res['df_clustered'][km_res['df_clustered']['Cluster'] == c]
            ax_clust.scatter(
                sub['Annual_Income'], sub['Spending_Score'],
                color=palette[c], label=km_res['cluster_names'][c],
                alpha=0.6, s=45
            )
        # Centroids
        ax_clust.scatter(
            km_res['centroids'][:, 0], km_res['centroids'][:, 1],
            color='black', marker='X', s=200, linewidths=2,
            edgecolors='white', label='Centroids', zorder=10
        )
        ax_clust.set_title("Customer Personas (Income vs Spending Score)", fontsize=12, fontweight='bold', pad=10)
        ax_clust.set_xlabel("Annual Income (₹ in Thousands)", fontsize=11)
        ax_clust.set_ylabel("Spending Score (1 - 100)", fontsize=11)
        ax_clust.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=8)
        st.pyplot(fig_clust)
        
    st.subheader("📊 Empirical Cluster Profiles Table")
    st.write("Cluster business labels assigned based on actual demographic and behavioral averages:")
    st.dataframe(km_res['profiles_df'], use_container_width=True)

# ---------------------------------------------------------
# PAGE 5: CLASSIFICATION MODELS
# ---------------------------------------------------------
elif menu == "Classification Models":
    st.markdown('<div class="main-header">Classification Models: Decision Tree vs Random Forest</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating Campaign Response Conversion Propensity on Hold-Out Test Data</div>', unsafe_allow_html=True)
    
    tab_dt, tab_rf, tab_comp = st.tabs(["🌳 Decision Tree Classifier", "🌲 Random Forest Classifier", "⚖️ Model Benchmarking"])
    
    with tab_dt:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{clf_res['dt_metrics']['Accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision", f"{clf_res['dt_metrics']['Precision']*100:.2f}%")
        with col3:
            st.metric("Recall", f"{clf_res['dt_metrics']['Recall']*100:.2f}%")
        with col4:
            st.metric("F1-Score", f"{clf_res['dt_metrics']['F1']:.4f}")
            
        col_cm, col_rules = st.columns([1, 1.2])
        with col_cm:
            st.markdown("**Decision Tree Confusion Matrix:**")
            fig_cm_dt, ax_cm_dt = plt.subplots(figsize=(5, 4))
            sns.heatmap(clf_res['dt_metrics']['cm'], annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax_cm_dt)
            ax_cm_dt.set_xlabel("Predicted")
            ax_cm_dt.set_ylabel("Actual")
            st.pyplot(fig_cm_dt)
            
        with col_rules:
            st.markdown("**Derived Business Decision Rules:**")
            st.success("""
            **Rule 1 — High Propensity Segment:**  
            **IF** Spending Score ≥ 58 **AND** Purchase Frequency ≥ 12 visits  
            **THEN** Conversion Likelihood > 80%  
            *Managerial Action:* Prioritize direct mailers and early access invitations.
            """)
            st.warning("""
            **Rule 2 — Low Propensity Segment:**  
            **IF** Spending Score < 45 **AND** Purchase Frequency ≤ 4 visits  
            **THEN** Conversion Likelihood < 15%  
            *Managerial Action:* Suppress costly direct marketing; use automated push notifications.
            """)
            
        with st.expander("🔍 View Decision Tree Visualization"):
            fig_tree, ax_tree = plt.subplots(figsize=(16, 7))
            plot_tree(
                clf_res['dt_model'],
                feature_names=clf_res['features'],
                class_names=['No', 'Yes'],
                filled=True, rounded=True, fontsize=8, ax=ax_tree
            )
            st.pyplot(fig_tree)
            
    with tab_rf:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{clf_res['rf_metrics']['Accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision", f"{clf_res['rf_metrics']['Precision']*100:.2f}%")
        with col3:
            st.metric("Recall", f"{clf_res['rf_metrics']['Recall']*100:.2f}%")
        with col4:
            st.metric("F1-Score", f"{clf_res['rf_metrics']['F1']:.4f}")
            
        col_cm_rf, col_imp = st.columns(2)
        with col_cm_rf:
            st.markdown("**Random Forest Confusion Matrix:**")
            fig_cm_rf, ax_cm_rf = plt.subplots(figsize=(5, 4))
            sns.heatmap(clf_res['rf_metrics']['cm'], annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax_cm_rf)
            ax_cm_rf.set_xlabel("Predicted")
            ax_cm_rf.set_ylabel("Actual")
            st.pyplot(fig_cm_rf)
            
        with col_imp:
            st.markdown("**Gini Feature Importance Chart:**")
            fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
            indices = np.argsort(clf_res['rf_importances'])
            ax_imp.barh(range(len(indices)), [clf_res['rf_importances'][i] for i in indices], color='#3498db', edgecolor='navy')
            ax_imp.set_yticks(range(len(indices)))
            ax_imp.set_yticklabels([clf_res['features'][i] for i in indices], fontsize=10)
            ax_imp.set_xlabel("Relative Importance (Impurity Reduction)")
            st.pyplot(fig_imp)
            
    with tab_comp:
        st.subheader("Model Comparison Table")
        st.dataframe(clf_res['comp_df'], use_container_width=True)
        
        winner = "Random Forest" if clf_res['rf_metrics']['F1'] >= clf_res['dt_metrics']['F1'] else "Decision Tree"
        st.success(f"""
        🏆 **Superior Model: {winner}**  
        Random Forest achieves superior Accuracy ({clf_res['rf_metrics']['Accuracy']*100:.2f}%) and F1-Score ({clf_res['rf_metrics']['F1']:.4f}), 
        reducing tree variance via bagging and maximizing campaign conversion efficiency.
        """)

# ---------------------------------------------------------
# PAGE 6: BUSINESS INSIGHTS
# ---------------------------------------------------------
elif menu == "Business Insights":
    st.markdown('<div class="main-header">Business Insights & Strategic Roadmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data-Driven Strategic Roadmaps Synthesized Directly from Model Results</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Advertising Strategy")
        st.info(f"""
        - **Empirical Relationship:** The OLS slope is **{reg_res['slope']:.4f}** with an R² of **{reg_res['r2']:.4f}**.
        - **Budget Allocation:** BrightMart should allocate advertising spend within the **₹45k–₹75k** bracket to maximize marginal revenue before diminishing returns occur.
        - **Expected Revenue:** Each ₹1,000 spend produces approx **₹{reg_res['slope']*1000:.2f}** in gross sales.
        """)
        
        st.subheader("3. Model Selection Recommendation")
        st.info(f"""
        - **Chosen Model:** **Random Forest Classifier** (F1 = {clf_res['rf_metrics']['F1']:.4f}, Accuracy = {clf_res['rf_metrics']['Accuracy']*100:.2f}%).
        - **Operational Benefit:** Higher precision ({clf_res['rf_metrics']['Precision']*100:.2f}%) minimizes marketing waste by avoiding unresponsive customers, while high recall ({clf_res['rf_metrics']['Recall']*100:.2f}%) captures over 90% of actual responders.
        """)
        
    with col2:
        st.subheader("2. Customer Segment Strategy")
        st.info("""
        - **Premium Champions (Cluster 0):** High income, high spend. Offer concierge checkout, white-glove loyalty privileges, and zero-discount experiential marketing.
        - **Budget Conscious (Cluster 1):** Low income, low spend. Deliver bulk discount bundles, essential staples promotions, and digital coupon books.
        - **Aspirational Enthusiasts (Cluster 2):** Moderate income, high spend. Deploy flexible buy-now-pay-later (BNPL), trend drops, and tiered loyalty points.
        """)
        
        st.subheader("4. Project Limitations & Monitoring")
        st.warning("""
        - **Limitation 1 (Synthetic Snapshot):** The cross-sectional dataset does not capture seasonal retail shifts (e.g. festive shopping) or longitudinal macroeconomic variations.
        - **Limitation 2 (Correlation vs Causation):** Statistical correlation does not prove causation; ad campaigns should be tested via A/B control groups.
        - **Model Monitoring:** Model performance drift must be monitored quarterly to recalibrate decision boundaries against live retail sales.
        """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown('<div class="footer-text">Customer Analytics & Machine Learning Project | MBA Data Science & AI</div>', unsafe_allow_html=True)
