import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title(" Week 4 Challenge: Interactive Data Dashboard")
st.write("Prepared by **Maryam Aslani**")

data = pd.read_csv("IBES.csv")

st.header("Variable Classification")
numeric_vars = data.select_dtypes(include=['number']).columns.tolist()
categorical_vars = data.select_dtypes(include=['object']).columns.tolist()
st.write("**Numeric Variables:**", numeric_vars)
st.write("**Categorical Variables:**", categorical_vars)

st.header("Handling Missing Data")
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

st.header("Interactive Outlier Detection and Handling")
numeric_cols = IBES_cleaned.select_dtypes(include='number').columns

selected_hist_var = st.selectbox("Select a variable for histogram:", numeric_cols)
fig, ax = plt.subplots()
ax.hist(IBES_cleaned[selected_hist_var].dropna(), bins=30, color='skyblue', edgecolor='black')
ax.set_title(f"Histogram of {selected_hist_var}")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
st.pyplot(fig)


selected_tickers = st.multiselect("Select TICKER(s) for boxplot:", IBES_cleaned['TICKER'].unique(), default=IBES_cleaned['TICKER'].unique())
filtered_data = IBES_cleaned[IBES_cleaned['TICKER'].isin(selected_tickers)]
st.subheader("Boxplot of ACTUAL Earnings by Company (Outliers Included)")
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.boxplot(x='TICKER', y='ACTUAL', data=filtered_data, ax=ax2)
ax2.set_title("Distribution of Actual Earnings by Company")
ax2.set_xlabel("Company Ticker")
ax2.set_ylabel("Actual Reported Earnings")
ax2.tick_params(axis='x', rotation=90)
plt.tight_layout()
st.pyplot(fig2)

st.header("Z-score Boxplot")
selected_norm_cols = st.multiselect("Select columns for Z-score boxplot:", numeric_cols, default=numeric_cols)
normalized_data = (IBES_cleaned[selected_norm_cols] - IBES_cleaned[selected_norm_cols].mean()) / IBES_cleaned[selected_norm_cols].std()
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=normalized_data, ax=ax3)
ax3.set_title("Boxplot of Standardized Numerical Columns")
ax3.set_ylabel("Standardized Value (Z-score)")
ax3.set_xlabel("Variables")
ax3.tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig3)

IBES_no_outliers = IBES_cleaned.copy()
for col in IBES_no_outliers.select_dtypes(include=['number']).columns:
    Q1 = IBES_no_outliers[col].quantile(0.25)
    Q3 = IBES_no_outliers[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    IBES_no_outliers = IBES_no_outliers[(IBES_no_outliers[col] >= lower_bound) & (IBES_no_outliers[col] <= upper_bound)]

st.header("Correlation Heatmap")
selected_corr_cols = st.multiselect("Select numeric columns for heatmap:", numeric_cols, default=numeric_cols)
corr_matrix = IBES_cleaned[selected_corr_cols].corr()
fig4, ax4 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={'label': 'Correlation'}, ax=ax4)
ax4.set_title("Correlation Heatmap of Selected Features")
plt.tight_layout()
st.pyplot(fig4)

st.header("Hypothesis 1: Testing the Hypothesis that Analyst Estimates (VALUE) Are Positively Correlated with Actual Earnings (ACTUAL")
show_reg_line = st.checkbox("Show regression line?", value=True)
fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.regplot(x='VALUE', y='ACTUAL', data=IBES_no_outliers, scatter_kws={'alpha': 0.3},
            line_kws={'color': 'red'} if show_reg_line else {}, ax=ax5)
ax5.set_title('Analyst Estimate vs Actual Earnings')
ax5.set_xlabel('Estimated Earnings (VALUE)')
ax5.set_ylabel('Actual Earnings (ACTUAL)')
ax5.grid(True)
plt.tight_layout()
st.pyplot(fig5)

st.header("Pairplot of Selected Variables")
selected_pairplot_cols = st.multiselect("Select columns for pairplot:", numeric_cols, default=numeric_cols[:3])
if len(selected_pairplot_cols) >= 2:
    pairplot_fig = sns.pairplot(IBES_cleaned[selected_pairplot_cols], corner=True)
    st.pyplot(pairplot_fig)
else:
    st.warning("Please select at least two variables for the pairplot.")

st.header("Hypothesis 2: The Accuracy of Analyst Estimates (VALUE) Has Improved Over Time — Average Forecast Error by Year")
IBES_no_outliers['YEAR'] = IBES_no_outliers['ACTDATS'].astype(str).str[:4].astype(int)
IBES_no_outliers['ERROR'] = ((IBES_no_outliers['VALUE'] - IBES_no_outliers['ACTUAL']) / IBES_no_outliers['ACTUAL']).abs()
avg_error_by_year = IBES_no_outliers.groupby('YEAR')['ERROR'].mean().reset_index()
min_year, max_year = int(avg_error_by_year['YEAR'].min()), int(avg_error_by_year['YEAR'].max())
year_range = st.slider("Select year range:", min_year, max_year, (min_year, max_year))
filtered_avg_error = avg_error_by_year[(avg_error_by_year['YEAR'] >= year_range[0]) & (avg_error_by_year['YEAR'] <= year_range[1])]
fig7, ax7 = plt.subplots(figsize=(10, 6))
ax7.plot(filtered_avg_error['YEAR'], filtered_avg_error['ERROR'], marker='o')
ax7.set_title('Average Forecast Error Over Time')
ax7.set_xlabel('Year')
ax7.set_ylabel('Average Error (|VALUE - ACTUAL|)')
ax7.grid(True)
plt.tight_layout()
st.pyplot(fig7)
