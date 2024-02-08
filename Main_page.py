import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
import pydeck as pdk 

#reference: https://www.youtube.com/watch?v=7yAw1nPareM
#reference for streamlit emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.set_page_config(
    page_title = "Consumer Behavior and Spending App",
    layout="wide"
)

st.title("Main Dashboard")
st.sidebar.success("Select a page.") 

shopping_df = pd.read_csv('shopping_behavior_new_updated.csv')

col1, col2 = st.columns((2))

# reference: https://www.youtube.com/watch?v=7yAw1nPareM
st.sidebar.header("Choose your filter:")

# Filters
with col1:
    st.subheader("Customers Gender")  
    gender = shopping_df['Gender'].value_counts().sort_index()
    plt.figure(figsize=(12, 8))

    gender.plot.bar(color=['pink', 'skyblue'])

    plt.xlabel("Gender")
    plt.ylabel("Counts")
    plt.title("Count by Gender")
    plt.xticks(rotation=90)
    plt.xticks(rotation=50, ha='right')

    st.pyplot(plt)
   

with col2:  
    st.write("Shipping Type")

# use pydeck for maping
