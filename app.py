import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Load Cleaned Data ===
st.title("📊 IBES Cleaned Data Dashboard")
data = pd.read_csv("IBES_cleaned.csv")

# === Identify numeric columns ===
numeric_cols = data.select_dtypes(include='number').columns.tolist()

# === Sidebar: Interactive Controls ===
st.sidebar.header("🔧 Controls")

# 1. Dropdown to select variable
selected_var = st.sidebar.selectbox("Select a numeric variable", numeric_cols)

# 2. Slider for histogram bins
bin_count = st.sidebar.slider("Number of bins", 5, 50, 20)

# 3. Checkbox for correlation heatmap
show_corr = st.sidebar.checkbox("Show Correlation Heatmap", value=True)

# 4. Multiselect for pairplot
pairplot_vars = st.sidebar.multiselect("Select variables for pairplot", numeric_cols, default=numeric_cols[:3])

# === Visualization 1: Histogram ===
st.subheader(f"📈 Histogram of {selected_var}")
fig1, ax1 = plt.subplots()
sns.histplot(data[selected_var].dropna(), bins=bin_count, kde=True, ax=ax1)
st.pyplot(fig1)

# === Visualization 2: Boxplot ===
st.subheader(f"📦 Boxplot of {selected_var}")
fig2, ax2 = plt.subplots()
sns.boxplot(y=data[selected_var].dropna(), ax=ax2)
st.pyplot(fig2)

# === Visualization 3: Correlation Heatmap ===
if show_corr:
    st.subheader("🧊 Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    corr = data[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

# === Visualization 4: Pairplot ===
if len(pairplot_vars) >= 2:
    st.subheader("🔁 Pairwise Scatter Plot")
    pairplot_fig = sns.pairplot(data[pairplot_vars], corner=True)
    st.pyplot(pairplot_fig)
else:
    st.warning("Please select at least two variables for the pairplot.")
