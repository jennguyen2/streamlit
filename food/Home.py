import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Create your shopping list",
    page_icon="🛒",
)

st.write("# Viet Foods! 🍜")
st.sidebar.success("Pick the Menu!")

# --- Add menu selection area ---
st.header("Pick Your Menu")

# Load the dataset (ensure required packages are installed)
df = pd.read_csv("hf://datasets/Hieu-Pham/kaggle_food_recipes/Food Ingredients and Recipe Dataset with Image Name Mapping.csv")
