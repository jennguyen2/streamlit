import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.title("Oneways")

# Sidebar: Data source
st.sidebar.header("Data Source")
use_default = st.sidebar.checkbox("Use default dataset")
default_df = None
if use_default:
    default_df = pd.read_csv("/Users/jennguyen/Data/dermatology.csv")
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "parquet"])

# S3 bucket link input
s3_link = st.sidebar.text_input("Or enter S3 bucket link (s3://...)", "")

df = None
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_parquet(uploaded_file)
elif s3_link.startswith("s3://"):
    import s3fs
    fs = s3fs.S3FileSystem()
    if s3_link.endswith(".csv"):
        with fs.open(s3_link, 'rb') as f:
            df = pd.read_csv(f)
    elif s3_link.endswith(".parquet"):
        with fs.open(s3_link, 'rb') as f:
            df = pd.read_parquet(f)
elif use_default and default_df is not None:
    df = default_df

if df is not None:
    st.sidebar.header("Oneway Analysis")
    target_col = st.sidebar.selectbox("Select target variable", df.columns)
    feature_cols = [col for col in df.columns if col != target_col]
    feature_col1 = st.sidebar.selectbox("Select first feature variable", feature_cols, key="feature1")
    feature_col2 = st.sidebar.selectbox("Select second feature variable", [col for col in feature_cols if col != feature_col1], key="feature2")

    # Facet variable selection
    facet_col = st.sidebar.selectbox(
        "Facet by (optional)", 
        ["None"] + feature_cols, 
        index=0
    )

    col1, col2 = st.columns(2)
    overall_target_mean = df[target_col].mean()

    def plot_oneway(feature_col, grouped, overall_target_mean, color, target_color):
        x_vals = grouped[feature_col].astype(str) if not pd.api.types.is_numeric_dtype(grouped[feature_col]) else grouped[feature_col]
        fig = go.Figure()
        fig.add_bar(
            x=x_vals,
            y=grouped['count'],
            name='Count',
            marker_color=color,
            yaxis='y1',
            showlegend=False
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=grouped['mean'],
                name=f"{target_col} Rate",
                mode='markers',
                marker_color=target_color,
                yaxis='y2',
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=[overall_target_mean] * len(grouped),
                name='Overall Mean Rate',
                mode='lines',
                line=dict(color=target_color, dash='dash'),
                yaxis='y2',
                showlegend=False
            )
        )
        fig.update_layout(
            xaxis_title=feature_col,
            yaxis=dict(
                title='Count',
                side='left'
            ),
            yaxis2=dict(
                title=f"{target_col} rate",
                overlaying='y',
                side='right',
                tickformat=".2%",
                range=[0, 1]
            ),
            showlegend=False,
            title=f"{feature_col} by {target_col} rate"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- First Oneway Plot ---
    with col1:
        grouped1 = df.groupby(feature_col1, dropna=False)[target_col].agg(['count', 'mean']).reset_index()
        grouped1 = grouped1.sort_values('count', ascending=False)
        plot_oneway(feature_col1, grouped1, overall_target_mean, 'lightblue', 'crimson')

    # --- Second Oneway Plot ---
    with col2:
        grouped2 = df.groupby(feature_col2, dropna=False)[target_col].agg(['count', 'mean']).reset_index()
        grouped2 = grouped2.sort_values('count', ascending=False)
        plot_oneway(feature_col2, grouped2, overall_target_mean, 'lightgreen', 'darkorange')

    # --- Facet Plot ---
    if facet_col != "None":
        st.header(f"Facet Oneway: {feature_col1} by {target_col} rate, faceted by {facet_col}")

        facet_df = df.copy()
        facet_df[feature_col1] = facet_df[feature_col1].astype(str)
        facet_df[facet_col] = facet_df[facet_col].astype(str)
        facet_values = facet_df[facet_col].unique()

        n_cols = 3
        n_rows = (len(facet_values) + n_cols - 1) // n_cols
        facet_grid = st.container()
        for row in range(n_rows):
            cols = facet_grid.columns(n_cols)
            for col_idx in range(n_cols):
                idx = row * n_cols + col_idx
                if idx >= len(facet_values):
                    break
                facet_value = facet_values[idx]
                sub_df = facet_df[facet_df[facet_col] == facet_value]
                grouped = sub_df.groupby(feature_col1, dropna=False)[target_col].agg(['count', 'mean']).reset_index()
                grouped = grouped.sort_values('count', ascending=False)
                with cols[col_idx]:
                    st.markdown(f"**{facet_col} = {facet_value}**")
                    plot_oneway(feature_col1, grouped, overall_target_mean, 'lightblue', 'crimson')

else:
    st.info("Please upload a CSV or Parquet file, enter an S3 link, or select the default dataset.")