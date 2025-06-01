import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Get Recipe",
    page_icon="📒",
)

st.write("# Recipes! 🍜")
st.sidebar.success("Pick the Menu!")

# Load the dataset (make sure required packages are installed: pandas, fsspec, s3fs, huggingface_hub)
df = pd.read_csv("hf://datasets/Hieu-Pham/kaggle_food_recipes/Food Ingredients and Recipe Dataset with Image Name Mapping.csv")

# Create a selectbox for recipe titles
selected_title = st.selectbox("Select a recipe:", df['Title'].dropna().unique())

# Display the ingredients for the selected recipe
ingredients = df.loc[df['Title'] == selected_title, 'Ingredients'].values
if ingredients.size > 0:
    st.subheader("Ingredients")
    st.write(ingredients[0])
else:
    st.write("No ingredients found for this recipe.")

# Display the ingredients for the selected recipe
recipes= df.loc[df['Title'] == selected_title, 'Instructions'].values
if recipes.size > 0:
    st.subheader("Recipe")
    st.write(recipes[0])
else:
    st.write("No recipe found.")