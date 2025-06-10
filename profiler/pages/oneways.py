import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Oneways")

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

if df is not None:
    st.sidebar.header("Oneway Analysis")
    target_col = st.sidebar.selectbox("Select target variable", df.columns)
    feature_cols = [col for col in df.columns if col != target_col]
    feature_col1 = st.sidebar.selectbox("Select first feature variable", feature_cols, key="feature1")
    feature_col2 = st.sidebar.selectbox("Select second feature variable", [col for col in feature_cols if col != feature_col1], key="feature2")

    col1, col2 = st.columns(2)
    overall_target_mean = df[target_col].mean()

    def plot_oneway(feature_col, grouped, overall_target_mean, color, target_color):
        # Convert categorical variables to string for plotting
        x_vals = grouped[feature_col].astype(str) if not pd.api.types.is_numeric_dtype(grouped[feature_col]) else grouped[feature_col]
        fig = go.Figure()
        fig.add_bar(
            x=x_vals,
            y=grouped['count'],
            name='Count',
            marker_color=color,
            yaxis='y1'
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=grouped['mean'],
                name=f"{target_col} Rate",
                mode='markers',
                marker_color=target_color,
                yaxis='y2'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=[overall_target_mean] * len(grouped),
                name='Overall Mean Rate',
                mode='lines',
                line=dict(color='gray', dash='dot'),
                yaxis='y2',
                showlegend=True
            )
        )
        fig.update_layout(
            xaxis_title=feature_col,
            yaxis=dict(
                title='Count',
                side='left'
            ),
            yaxis2=dict(
                title=f"{target_col} Rate",
                overlaying='y',
                side='right',
                tickformat=".2%",
                range=[0, 1]
            ),
            legend=dict(x=0.01, y=0.99),
            title=f"{feature_col} by {target_col} Rate"
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

else:
    st.info("Please upload a CSV or Parquet file or select the default dataset.")