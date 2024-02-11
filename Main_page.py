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

st.title(":bar_chart: Consumer Behavior and Spending Dashboard")
st.sidebar.success("Select a page.") 

shopping_df = pd.read_csv('shopping_behavior_new_updated.csv')

col1, col2 = st.columns((2))

# reference: https://www.youtube.com/watch?v=7yAw1nPareM
st.sidebar.header("Choose your filter:")

#Top KPI's
total_sales = int(shopping_df['Purchase Amount (USD)'].sum())
average_rating = round(shopping_df['Review Rating'].mean(), 1)
star_rating = ":star:" * int(round(average_rating, 0))

# Filters
with col1:
    st.subheader("Total Sales:")
    st.subheader(f"US $ {total_sales:,}")
    st.markdown("---")
    
    # reference: https://www.youtube.com/watch?v=Sb0A9i6d320&list=PLHgX2IExbFovFg4DI0_b3EWyIGk-oGRzq
    fig_gender = px.pie(shopping_df, values="Purchase Amount (USD)", names="Gender", title="Customer's Gender", hole= 0.5)
    fig_gender.update_traces(text = shopping_df['Gender'], textposition = "outside")
    
with col2:  
    st.subheader("Average Rating:")
    st.subheader(f"{average_rating} {star_rating}")
    st.markdown("---")
    
    options_top_location = st.selectbox('Select:', ['Top 5 Location', 'Top 10 Location'])
    
    if options_top_location == 'Top 5 Location':
        top_location = shopping_df['Location'].value_counts().head(5)
        top_location.sort_values(ascending=False, inplace=True)
    else:
        top_location = shopping_df['Location'].value_counts().head(10)
        top_location.sort_values(ascending=False, inplace=True)
    
    fig_location = px.bar(
        top_location,
        x=top_location.values,
        y=top_location.index,
        orientation="h",
        title="<b>Top Location</b>",
        color="Location",
        template="plotly_white",
    )
        
    fig_category = px.bar(
        shopping_df,
        x="Category",
        orientation="v",
        title="<b>Popular Category</b>",
        color="Category",
        template="plotly_white",
    )
    
col2.plotly_chart(fig_location, use_container_width=True)
col2.plotly_chart(fig_category, use_container_width=True)
col1.plotly_chart(fig_gender, use_container_width=True)

# use pydeck for maping
