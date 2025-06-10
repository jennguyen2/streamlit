import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Oneway-Static")

# Sidebar: Data source
st.sidebar.header("Data Source")
use_default = st.sidebar.checkbox("Use default dataset")
default_df = None
if use_default:
    default_df = pd.read_csv("/Users/jennguyen/Data/dermatology.csv")
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "parquet"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_parquet(uploaded_file)
elif use_default and default_df is not None:
    df = default_df
else:
    df = None

# --- Add filters in the sidebar ---
if df is not None:
    st.sidebar.header("🔎 Data Filters")
    filter_vars = st.sidebar.multiselect(
        "Select columns to filter",
        options=df.columns.tolist(),
        help="Choose which columns you want to filter."
    )
    filtered_df = df.copy()
    for col in filter_vars:
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val, max_val = float(df[col].min()), float(df[col].max())
            selected = st.sidebar.slider(
                f"{col} range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
            filtered_df = filtered_df[(filtered_df[col] >= selected[0]) & (filtered_df[col] <= selected[1])]
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            unique_vals = df[col].dropna().unique()
            selected = st.sidebar.multiselect(f"{col} values", unique_vals, default=list(unique_vals))
            filtered_df = filtered_df[filtered_df[col].isin(selected)]
    df = filtered_df

if df is not None:
    st.sidebar.header("Target")
    target_col = st.sidebar.selectbox("Select target variable", df.columns)
    feature_cols = [col for col in df.columns if col != target_col]
    overall_target_mean = df[target_col].mean()

    # Precompute groupings for all features
    groupings = {}
    for feature_col in feature_cols:
        grouped = df.groupby(feature_col, dropna=False)[target_col].agg(['count', 'mean']).reset_index()
        if not pd.api.types.is_numeric_dtype(df[feature_col]):
            grouped = grouped.sort_values('count', ascending=False)
        else:
            grouped = grouped.sort_values(feature_col)
        groupings[feature_col] = grouped

    def plot_oneway_static(feature_col, grouped, overall_target_mean):
        x_vals = grouped[feature_col].astype(str) if not pd.api.types.is_numeric_dtype(grouped[feature_col]) else grouped[feature_col]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = ax1.twinx()

        # Bar plot for counts (light blue)
        bars = sns.barplot(x=x_vals, y=grouped['count'], color='lightblue', ax=ax1)
        ax1.set_ylabel('Count')
        ax1.set_xlabel(feature_col)
        ax1.tick_params(axis='x', rotation=45)

        # Add count labels near the top of bars (not above)
        heights = [bar.get_height() for bar in bars.patches]
        y_max = max(heights) if heights else 0
        margin = -max(y_max * 0.03, 2)
        for bar, count in zip(bars.patches, grouped['count']):
            height = bar.get_height()
            ax1.annotate(
                f"{int(count)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, margin),
                textcoords="offset points",
                ha='center',
                va='top',
                fontsize=9,
                color='black',
                clip_on=False
            )

        claret = '#7B1F3A'
        ax2.scatter(x=x_vals, y=grouped['mean'], color=claret, label=f"{target_col} Rate", zorder=5)
        ax2.axhline(y=overall_target_mean, color=claret, linestyle='dotted', linewidth=2)
        ax2.set_ylabel(f"{target_col} Rate", color=claret)
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', colors=claret)
        if ax2.get_legend():
            ax2.legend_.remove()

        plt.title(f"{feature_col} by {target_col} Rate")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    n = 3
    for i in range(0, len(feature_cols), n):
        cols = st.columns(n)
        for j, feature_col in enumerate(feature_cols[i:i+n]):
            with cols[j]:
                plot_oneway_static(
                    feature_col,
                    groupings[feature_col],
                    overall_target_mean
                )

else:
    st.info("Please upload a CSV or Parquet file or select the default dataset.")

