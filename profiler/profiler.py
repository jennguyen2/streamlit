import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Config
st.set_page_config(
    page_title="Interactive EDA",
    page_icon="📊",
    layout="wide"
)

# --- Page selector ---
page = st.sidebar.selectbox(
    "Select Page",
    ["EDA Dashboard", "Oneways", "Oneways Static"]
)

if page == "Oneways":
    # Import and run the custom plot page
    import importlib.util
    import sys
    import os

    oneways_path = os.path.join(os.path.dirname(__file__), "pages", "oneways.py")
    spec = importlib.util.spec_from_file_location("oneways", oneways_path)
    oneways = importlib.util.module_from_spec(spec)
    sys.modules["oneways"] = oneways
    spec.loader.exec_module(oneways)
    st.stop()  # Stop further execution of this file
    
if page == "Oneways Static":
    # Import and run the static oneways page
    import importlib.util
    import sys
    import os

    oneways_static_path = os.path.join(os.path.dirname(__file__), "pages", "oneways_static.py")
    spec = importlib.util.spec_from_file_location("oneways_static", oneways_static_path)
    oneways_static = importlib.util.module_from_spec(spec)
    sys.modules["oneways_static"] = oneways_static
    spec.loader.exec_module(oneways_static)
    st.stop()  # Stop further execution of this file

st.title("📊 Interactive EDA Dashboard")
st.markdown("Upload a CSV or Parquet file to get started.")

# Sidebar: Option to use a default dataset
st.sidebar.header("Data Source")
use_default = st.sidebar.checkbox("Use default dataset")

default_df = None
if use_default:
    # Replace this with your actual default dataset path or loading logic
    default_df = pd.read_csv("/Users/jennguyen/Data/dermatology.csv")

# File upload
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "parquet"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_parquet(uploaded_file)
elif use_default and default_df is not None:
    df = default_df
else:
    df = None

if df is not None:
    # 🛠 Ensure column names are unique
    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    for col in cols:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("🔍 Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    # Show missing counts/percentage and data types in one DataFrame
    overview_df = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(overview_df)

    # Sidebar for EDA options
    st.sidebar.header("EDA Options")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    st.sidebar.subheader("📈 Univariate Analysis")
    col_to_plot = st.sidebar.selectbox("Select a column", df.columns)

    if col_to_plot:
        st.subheader(f"Distribution: {col_to_plot}")
        if df[col_to_plot].dtype in [np.float64, np.int64]:
            fig = px.histogram(df, x=col_to_plot, nbins=30, title=f"Histogram of {col_to_plot}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            counts = df[col_to_plot].value_counts().nlargest(20)
            fig = px.bar(
                x=counts.index,
                y=counts.values,
                labels={'x': col_to_plot, 'y': 'Count'},
                title=f"Top 20 Values in {col_to_plot}"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.sidebar.subheader("📉 Bivariate Analysis")
    x_axis = st.sidebar.selectbox("X-axis", numeric_cols, key="x_axis")
    y_axis = st.sidebar.selectbox("Y-axis", numeric_cols, key="y_axis")

    if x_axis and y_axis:
        st.subheader(f"Bivariate Analysis: {x_axis} vs {y_axis}")

        # Prepare data for plotting
        grouped = df.groupby(x_axis)[y_axis].agg(['count', 'mean']).reset_index()

        fig = px.bar(
            grouped,
            x=x_axis,
            y='count',
            labels={x_axis: x_axis, 'count': f'Count of {y_axis}'},
            title=f"Counts of {y_axis} by {x_axis}"
        )

        # Add line for mean overlay
        fig.add_scatter(
            x=grouped[x_axis],
            y=grouped['mean'],
            mode='lines+markers',
            name=f"Mean {y_axis}",
            yaxis="y2"
        )

        # Set up secondary y-axis
        fig.update_layout(
            yaxis2=dict(
                title=f"Mean {y_axis}",
                overlaying='y',
                side='right'
            ),
            legend=dict(x=0.01, y=0.99)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.sidebar.subheader("🔗 Correlation Matrix")
    show_corr = st.sidebar.checkbox("Show Correlation Heatmap")
    corr_threshold = st.sidebar.slider(
        "Correlation threshold (absolute value)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Show only correlations above this threshold"
    )

    if show_corr and numeric_cols:
        st.subheader("📌 Correlation Heatmap")
        corr = df[numeric_cols].corr()

        # Mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Apply threshold
        corr_masked = corr.copy()
        corr_masked[mask] = np.nan
        corr_masked = corr_masked.where(corr_masked.abs() >= corr_threshold)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            corr_masked,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            mask=mask,
            ax=ax,
            cbar=True,
            vmin=-1, vmax=1,
            linewidths=0.5,
            linecolor='gray'
        )
        ax.set_facecolor('white')  # Set white background for the plot area
        fig.patch.set_facecolor('white')  # Set white background for the figure
        st.pyplot(fig)

    # --- Add filters in the sidebar ---
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
else:
    st.info("Please upload a CSV or Parquet file or select the default dataset.")

