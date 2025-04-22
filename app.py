import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Week 4 Challenge: Interactive Data Dashboard")
st.write("Prepared by **Maryam Aslani**")

# === Load Data ===
data = pd.read_csv("IBES.csv")

# === TASK 1: Variable Classification ===
st.header("Task 1: Variable Classification")

numeric_vars = data.select_dtypes(include=['number']).columns.tolist()
categorical_vars = data.select_dtypes(include=['object']).columns.tolist()

st.write("**Numeric Variables:**", numeric_vars)
st.write("**Categorical Variables:**", categorical_vars)

# === TASK 2: Handling Missing Data ===
st.header("Task 2: Handling Missing Data")

missing_counts = data.isnull().sum()
st.write("**Missing values per column:**")
st.write(missing_counts)

IBES_cleaned = data.copy()
for column in IBES_cleaned.columns:
    missing_count = IBES_cleaned[column].isnull().sum()
    missing_pct = (missing_count / len(IBES_cleaned)) * 100
    if missing_count == 0:
        continue
    elif missing_pct <= 30:
        if IBES_cleaned[column].dtype in ['float64', 'int64']:
            IBES_cleaned[column] = IBES_cleaned[column].fillna(IBES_cleaned[column].mean())
        else:
            IBES_cleaned[column] = IBES_cleaned[column].fillna(IBES_cleaned[column].mode()[0])
    else:
        IBES_cleaned.drop(columns=[column], inplace=True)

st.write("**Remaining columns after cleaning:**")
st.write(IBES_cleaned.columns.tolist())

# === TASK 3: Outlier Detection and Handling ===
st.header("Task 3: Outlier Detection and Handling")

numeric_cols = IBES_cleaned.select_dtypes(include='number').columns
st.subheader("Histograms of All Numerical Variables")
num_vars = len(numeric_cols)
cols = 3
rows = (num_vars + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    axes[i].hist(IBES_cleaned[col].dropna(), bins=30, color='skyblue', edgecolor='black')
    axes[i].set_title(col)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
st.pyplot(fig)

st.subheader("Boxplot of ACTUAL Earnings by Company (Outliers Included)")
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.boxplot(x='TICKER', y='ACTUAL', data=IBES_cleaned, ax=ax2)
ax2.set_title("Distribution of Actual Earnings by Company")
ax2.set_xlabel("Company Ticker")
ax2.set_ylabel("Actual Reported Earnings")
ax2.tick_params(axis='x', rotation=90)
plt.tight_layout()
st.pyplot(fig2)

# === Z-score Normalization Boxplot ===
st.subheader("Boxplot of Standardized Numerical Columns")
normalized_data = (IBES_cleaned[numeric_cols] - IBES_cleaned[numeric_cols].mean()) / IBES_cleaned[numeric_cols].std()
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=normalized_data, ax=ax3)
ax3.set_title("Boxplot of Standardized Numerical Columns")
ax3.set_ylabel("Standardized Value (Z-score)")
ax3.set_xlabel("Variables")
ax3.tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig3)

# === Remove Outliers ===
IBES_no_outliers = IBES_cleaned.copy()
for col in IBES_no_outliers.select_dtypes(include=['number']).columns:
    Q1 = IBES_no_outliers[col].quantile(0.25)
    Q3 = IBES_no_outliers[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    IBES_no_outliers = IBES_no_outliers[(IBES_no_outliers[col] >= lower_bound) & (IBES_no_outliers[col] <= upper_bound)]

# === Correlation Heatmap ===
st.header("Correlation Heatmap of Numerical Features")
corr_matrix = IBES_cleaned[numeric_cols].corr()
fig4, ax4 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={'label': 'Correlation'}, ax=ax4)
ax4.set_title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
st.pyplot(fig4)

# === Regression Plot (Hypothesis 1) ===
st.header("Hypothesis 1: VALUE is Positively Correlated with ACTUAL")
fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.regplot(x='VALUE', y='ACTUAL', data=IBES_no_outliers, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'}, ax=ax5)
ax5.set_title('Analyst Estimate vs Actual Earnings')
ax5.set_xlabel('Estimated Earnings (VALUE)')
ax5.set_ylabel('Actual Earnings (ACTUAL)')
ax5.grid(True)
plt.tight_layout()
st.pyplot(fig5)

# === Pairplot ===
st.header("Pairwise Scatter Plots of Numerical Variables")
fig6 = sns.pairplot(IBES_cleaned[numeric_cols], corner=True)
st.pyplot(fig6)

# === Forecast Error Over Time (Hypothesis 3) ===
st.header("Hypothesis 3: Forecast Error Improvement Over Time")
IBES_no_outliers['YEAR'] = IBES_no_outliers['ACTDATS'].astype(str).str[:4].astype(int)
IBES_no_outliers['ERROR'] = ((IBES_no_outliers['VALUE'] - IBES_no_outliers['ACTUAL']) / IBES_no_outliers['ACTUAL']).abs()
avg_error_by_year = IBES_no_outliers.groupby('YEAR')['ERROR'].mean().reset_index()

fig7, ax7 = plt.subplots(figsize=(10, 6))
ax7.plot(avg_error_by_year['YEAR'], avg_error_by_year['ERROR'], marker='o')
ax7.set_title('Average Forecast Error Over Time')
ax7.set_xlabel('Year')
ax7.set_ylabel('Average Error (|VALUE - ACTUAL|)')
ax7.grid(True)
plt.tight_layout()
st.pyplot(fig7)
