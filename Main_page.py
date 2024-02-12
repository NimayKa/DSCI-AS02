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

# col1, col2 = st.columns((2.5, 2))
col1, col2 = st.columns((2.5, 3.5), gap='medium')
rows = st.columns((2))
tab1, tab2 = st.tabs(["📈 Chart", "🗃 Popularity Chart"])
# reference: https://docs.streamlit.io/library/api-reference/layout/st.tabs

# reference: https://www.youtube.com/watch?v=7yAw1nPareM
    
with col1:
    #Top KPI's
    total_sales = int(shopping_df['Purchase Amount (USD)'].sum())
    average_rating = round(shopping_df['Review Rating'].mean(), 1)
    star_rating = ":star:" * int(round(average_rating, 0))

    # Filters

    st.subheader("Total Sales:")
    st.subheader(f"US $ {total_sales:,}")
    
with col2:
    st.subheader("Average Rating:")
    st.subheader(f"{average_rating} {star_rating}")  

# reference: https://docs.streamlit.io/library/api-reference/layout/st.tabs
with tab1:
    col2, col3 = st.columns((1, 5), gap='medium')
    # reference: https://www.youtube.com/watch?v=Sb0A9i6d320&list=PLHgX2IExbFovFg4DI0_b3EWyIGk-oGRzq
    with col2:
        st.header("Choose your filter:")
        options_by_purchase = st.radio('Promo Code Used:', ('None','Yes', 'No'))
        options_by_fop = st.selectbox('Select:', shopping_df['Frequency of Purchases'].unique())

    with col3:
        if options_by_purchase == 'Yes':
            filtered = shopping_df[shopping_df['Promo Code Used'] == 'Yes']
        elif options_by_purchase == 'No':
            filtered = shopping_df[shopping_df['Promo Code Used'] == 'No']
        else:
                filtered = shopping_df.groupby('Gender')['Purchase Amount (USD)'].sum().reset_index()
            
        if options_by_fop == 'Every 3 Months':
            filtered = shopping_df[shopping_df['Frequency of Purchases'] == 'Every 3 Months']
        elif options_by_fop== 'Annually':
            filtered = shopping_df[shopping_df['Frequency of Purchases'] == 'Annually']
        elif options_by_fop== 'Quarterly':
            filtered = shopping_df[shopping_df['Frequency of Purchases'] == 'Quarterly']
        elif options_by_fop== 'Monthly':
            filtered = shopping_df[shopping_df['Frequency of Purchases'] == 'Monthly']
        elif options_by_fop== 'Bi-Weekly':
            filtered = shopping_df[shopping_df['Frequency of Purchases'] == 'Bi-Weekly']
        elif options_by_fop== 'Fortnightly':
            filtered = shopping_df[shopping_df['Frequency of Purchases'] == 'Fortnightly']
        elif options_by_fop== 'Weekly':
            filtered = shopping_df[shopping_df['Frequency of Purchases'] == 'Weekly']
        else:
            filtered = shopping_df.groupby('Gender')['Purchase Amount (USD)'].sum().reset_index()
                
    # reference: https://plotly.com/python/discrete-color/
        fig_gender = px.pie(filtered, values="Purchase Amount (USD)", names="Gender", color_discrete_sequence=px.colors.qualitative.Plotly, title="Customer's Gender", hole= 0.5)
        fig_gender.update_traces(text = filtered['Gender'], textposition = "outside")
            
        st.plotly_chart(fig_gender, use_container_width=True)
            
        season = shopping_df.groupby('Season')['Purchase Amount (USD)'].sum().sort_values(ascending=False)
        fig_season = px.line(season,
                                x=season.index,
                                y="Purchase Amount (USD)",
                                title="Seasons by Revenue",
                                template="plotly_white",
                                markers=True)
            
        st.plotly_chart(fig_season, use_container_width=True)
     
with tab2: 
    col4, col5 = st.columns((1, 5), gap='medium')
    
    with col4:
        st.header("Choose your filter:")
        options_top_location = st.selectbox('Select:', ['Top 5 Location', 'Top 10 Location'])
    
    with col5:
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
                
        st.plotly_chart(fig_location, use_container_width=True)
                
        fig_category = px.bar(
                shopping_df,
                x="Category",
                orientation="v",
                title="<b>Popular Category</b>",
                color="Category",
                template="plotly_white",
            )
            
        st.plotly_chart(fig_category, use_container_width=True)

  # reference: https://plotly.com/python/line-charts/

# use pydeck for maping

