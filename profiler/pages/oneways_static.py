import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_pdf import PdfPages
import base64

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

    # Option to export
    export_format = st.sidebar.selectbox("Export Oneways as", ["None", "PDF", "HTML"])

    # Store figures for export
    figures = []
    feature_titles = []

    def plot_oneway_static(feature_col, grouped, overall_target_mean, collect_figs=False):
        x_vals = grouped[feature_col].astype(str) if not pd.api.types.is_numeric_dtype(grouped[feature_col]) else grouped[feature_col]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = ax1.twinx()

        # Bar plot for counts (light blue)
        bars = sns.barplot(x=x_vals, y=grouped['count'], color='lightblue', ax=ax1)
        ax1.set_ylabel('Count')
        ax1.set_xlabel(feature_col)
        ax1.tick_params(axis='x', rotation=45)

        # Add count labels on top of bars, with margin from the top
        y_max = max([bar.get_height() for bar in bars.patches]) if bars.patches else 0
        for bar, count in zip(bars.patches, grouped['count']):
            height = bar.get_height()
            margin = max(y_max * 0.02, 2)
            ax1.annotate(
                f"{int(count)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, margin),
                textcoords="offset points",
                ha='center',
                va='top',
                fontsize=9,
                clip_on=False
            )

        claret = '#7B1F3A'

        # Overlay: scatter for target rate (claret)
        ax2.scatter(x=x_vals, y=grouped['mean'], color=claret, label=f"{target_col} Rate", zorder=5)

        # Dotted reference line for overall mean (claret, covers whole y-axis)
        ax2.axhline(y=overall_target_mean, color=claret, linestyle='dotted', linewidth=2)
        ax2.set_ylabel(f"{target_col} Rate", color=claret)
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', colors=claret)
        # Remove legend
        ax2.legend_.remove() if ax2.get_legend() else None

        plt.title(f"{feature_col} by {target_col} Rate")
        plt.tight_layout()
        st.pyplot(fig)
        if collect_figs:
            figures.append(fig)
            feature_titles.append(feature_col)
        else:
            plt.close(fig)

    n = 3
    for i in range(0, len(feature_cols), n):
        cols = st.columns(n)
        for j, feature_col in enumerate(feature_cols[i:i+n]):
            with cols[j]:
                grouped = df.groupby(feature_col, dropna=False)[target_col].agg(['count', 'mean']).reset_index()
                # Sort by counts if categorical, else by feature value
                if not pd.api.types.is_numeric_dtype(df[feature_col]):
                    grouped = grouped.sort_values('count', ascending=False)
                else:
                    grouped = grouped.sort_values(feature_col)
                plot_oneway_static(
                    feature_col,
                    grouped,
                    overall_target_mean,
                    collect_figs=(export_format != "None")
                )

    # Export logic
    if export_format == "PDF" and figures:
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            # Add a title page
            plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            plt.text(0.5, 0.7, "Oneway Analysis Report", fontsize=24, ha='center')
            plt.text(0.5, 0.6, f"Target Variable: {target_col}", fontsize=16, ha='center')
            plt.text(0.5, 0.55, f"Number of Features: {len(feature_cols)}", fontsize=12, ha='center')
            plt.text(0.5, 0.5, "Generated by Streamlit", fontsize=10, ha='center')
            pdf.savefig()
            plt.close()
            # Add each row of 3 plots per page
            for i in range(0, len(figures), n):
                fig_row = figures[i:i+n]
                titles_row = feature_titles[i:i+n]
                fig, axs = plt.subplots(1, len(fig_row), figsize=(6 * len(fig_row), 4))
                if len(fig_row) == 1:
                    axs = [axs]
                for ax, subfig, title in zip(axs, fig_row, titles_row):
                    # Draw the saved figure onto the subplot
                    subfig_axes = subfig.get_axes()
                    for sub_ax in subfig_axes:
                        for artist in sub_ax.get_children():
                            try:
                                artist.figure = fig
                                ax._add_text(artist)
                            except Exception:
                                pass
                    ax.imshow(subfig.canvas.buffer_rgba())
                    ax.axis('off')
                    ax.set_title(f"{title} by {target_col} Rate", fontsize=12)
                    plt.close(subfig)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        st.sidebar.download_button(
            label="Download Oneways PDF Report",
            data=pdf_buffer.getvalue(),
            file_name="oneways_report.pdf",
            mime="application/pdf"
        )

    elif export_format == "HTML" and figures:
        html_buffer = io.StringIO()
        # Add a title and summary at the top of the HTML
        html_buffer.write(f"<h1>Oneway Analysis Report</h1>")
        html_buffer.write(f"<h2>Target Variable: {target_col}</h2>")
        html_buffer.write(f"<h3>Number of Features: {len(feature_cols)}</h3>")
        html_buffer.write("<hr>")
        # Add 3 plots per row
        for i in range(0, len(figures), n):
            html_buffer.write('<div style="display: flex; flex-wrap: wrap;">')
            for fig, feature_col in zip(figures[i:i+n], feature_titles[i:i+n]):
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format='png', bbox_inches='tight')
                img_buf.seek(0)
                img_b64 = base64.b64encode(img_buf.read()).decode()
                html_buffer.write(
                    f'<div style="flex: 1; min-width: 300px; margin: 10px;">'
                    f'<h4>{feature_col} by {target_col} Rate</h4>'
                    f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;"><br><br>'
                    f'</div>'
                )
                plt.close(fig)
            html_buffer.write('</div><hr>')
        html_content = html_buffer.getvalue()
        st.sidebar.download_button(
            label="Download Oneways HTML Report",
            data=html_content,
            file_name="oneways_report.html",
            mime="text/html"
        )

else:
    st.info("Please upload a CSV or Parquet file or select the default dataset.")

