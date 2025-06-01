import streamlit as st

st.set_page_config(
    page_title="To Do for App"
    # page_icon="📝",
)

st.write("# Backlog: add new features")

st.write("### To Do List for the App")
st.checkbox("add a search bar")
st.checkbox("add categories for recipes: soup, appetizer, dessert, etc.")
st.checkbox("reformat ingredients")

st.write("#### Personalized recipes")
st.checkbox("home recipes")


st.write("### Nice to have")
st.checkbox("dropdown for dates")
st.checkbox("add images to recipes")

st.write("### Where to buy? Best deals, coupons, etc.")