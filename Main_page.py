import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
import pydeck as pdk 

st.set_page_config(
    page_title = "Consumer Behavior and Shopping Habits",
    layout="wide"
)

st.title(":bar_chart: Consumer Behavior and Spending Dashboard")
st.sidebar.success("Select a page.") 

shopping_df = pd.read_csv('shopping_behavior_new_updated.csv')

col1, col2, col3 = st.columns((2.5, 2.5, 2.5), gap='medium')
    
with col1:
    #Top KPI's
    total_sales = int(shopping_df['Purchase Amount (USD)'].sum())
    average_rating = round(shopping_df['Review Rating'].mean(), 1)
    star_rating = ":star:" * int(round(average_rating, 0))
    average_age = round(shopping_df['Age'].mean())
    
    # Filters
    st.subheader("Total Sales:")
    st.subheader(f"US $ {total_sales:,}")
    st.markdown('__________')
    
with col2:
    st.subheader("Average Rating:")
    st.subheader(f"{average_rating} {star_rating}")  
    st.markdown('___________')

with col3:
    st.subheader("Average Age:")
    st.subheader(f"{average_age} years")  
    st.markdown('___________')

cols1, cols2, cols3= st.columns((0.8,2,3), gap='medium')

with cols1:
        st.write("Filter: Customer's Gender Chart")
        slider_age = st.slider('Age slider', min_value=shopping_df['Age'].min(), max_value=shopping_df['Age'].max())
         
        if slider_age <= 19:
            st.write("Age Group: Teenager")
            filtered = shopping_df[shopping_df['Age'] == slider_age]
        elif slider_age >= 20 and slider_age <= 24:
            st.write("Age Group: Young Adult")
            filtered = shopping_df.groupby('Gender')['Purchase Amount (USD)'].sum().reset_index()
        elif slider_age >= 25 and slider_age <= 49:
            st.write("Age Group: Adult")
        else:
            st.write("Age Group: Old")
        
        options_by_fop = st.selectbox('Select:', shopping_df['Frequency of Purchases'].unique(), index=None )
        st.markdown("-------")
        
        st.write("Filter: Seasons by Total Sales")
        options_by_discount = st.selectbox('Select Category:', shopping_df['Discount Applied'].unique(), index=None)
        
        
with cols2:
        st.subheader("Customer's Gender", divider='rainbow')
            
        if options_by_fop:
            filtered = shopping_df[shopping_df['Frequency of Purchases'] == options_by_fop]
        else:
            filtered = shopping_df.groupby('Gender')['Purchase Amount (USD)'].sum().reset_index()
            
          
        fig_gender = px.pie(filtered, values="Purchase Amount (USD)", names="Gender", color_discrete_sequence=px.colors.qualitative.Plotly, hole= 0.5)
        fig_gender.update_traces(text = filtered['Gender'], textposition = "outside")
                
        st.plotly_chart(fig_gender, use_container_width=True)
             
with cols3:
        # Line chart
        st.subheader("Seasons by Total Sales", divider='rainbow')
        
        if options_by_discount:
            season = shopping_df[shopping_df['Discount Applied'] == options_by_discount]
        else:
            season = shopping_df.groupby('Season')['Purchase Amount (USD)'].sum().sort_values(ascending=False)
            
        fig_season = px.line(season,
                             x=season.index,
                             y='Purchase Amount (USD)',
                            template="plotly_white",
                            markers=True)
                
        st.plotly_chart(fig_season, use_container_width=True)
    
col1s, col2s, col3s= st.columns((0.9,2,3), gap='medium')
    
with col1s:
    
        st.markdown("                                  ")
        st.markdown("-------")
        st.markdown("Filter: Top Location among Customers chart")
        options_top_location = st.selectbox('Select:', ['Top 5 Location', 'Top 10 Location'])
        # options_by_subscription = st.selectbox('Select Subscription Status:', shopping_df['Subscription Status'].unique(), index=None)
        st.markdown("-------")
        st.markdown("Filter: Popular Category chart")
        options_by_age = st.selectbox('Select age group:', shopping_df['Age Group'].unique(), index=None)
        options_by_gender = st.radio('Select gender:', ['None', 'Female', 'Male'])
        st.markdown('___________')
        
    
with col2s:
    st.markdown("                                  ")
    st.subheader("Top Location among Customers", divider='rainbow')
        
    if options_top_location == 'Top 5 Location':
        top_location = shopping_df['Location'].value_counts().head(5)
        top_location.sort_values(ascending=False, inplace=True)
    else:
        top_location = shopping_df['Location'].value_counts().head(10)
        top_location.sort_values(ascending=False, inplace=True)
    
    # if options_by_subscription:
        # top_location =  shopping_df[shopping_df['Subscription Status'] == options_by_subscription]
    
    fig_location = px.bar(
                    top_location,
                    x=top_location.values,
                    y=top_location.index,
                    orientation="h",
                    color="Location",
                    template="plotly_white",
                    labels={'x':'Count', 'y': 'Locations'}
                )
        
        # add filters according revenue (selectboxx)
    st.plotly_chart(fig_location, use_container_width=True)
        
        
with col3s:
    
    st.markdown("                                  ")    
    st.subheader("Popular Category", divider='rainbow')
           
    if options_by_gender == 'Female':
        popular = shopping_df[shopping_df['Gender'] == 'Female']
    elif options_by_gender == 'Male':
        popular = shopping_df[shopping_df['Gender'] == 'Male']
    else:
        popular = shopping_df.groupby('Category').count().reset_index()
    
    popular = popular.rename(columns={"Customer ID": "Count"})
            
    fig_category = px.bar(
                popular,
                x="Category",
                y="Count",
                orientation="v",
                color="Category",
                template="plotly",)
            
            
    st.plotly_chart(fig_category, use_container_width=True)

st.subheader("Popular Shipping Type", divider='rainbow')    
options_seasons = st.selectbox('Select season:', shopping_df['Season'].unique(), index=None)

if options_seasons:
    shipmode = shopping_df[shopping_df['Season'] == options_seasons].groupby('Shipping Type')['Purchase Amount (USD)'].sum().reset_index()
else:
    shipmode = shopping_df.groupby('Shipping Type')['Purchase Amount (USD)'].sum().reset_index()

fig_shipping = px.bar(
                shipmode,
                x="Shipping Type",
                y="Purchase Amount (USD)",
                orientation="v",
                color="Shipping Type",
                template="plotly_white",
                labels={'x':'Shipping Type', 'y': 'PA(USD)'},
                text= shipmode['Purchase Amount (USD)'].apply(lambda x: f'{x/1000:.1f}')
                )
st.plotly_chart(fig_shipping, use_container_width=True)
st.markdown("                                    ")

st.subheader("Purchase Amount and Frequency of Purchases", divider='rainbow')  
fig_box = px.box(shopping_df,
                         x='Frequency of Purchases',
                         y='Purchase Amount (USD)',
                         width= 1600,
                         height= 500)


st.plotly_chart(fig_box)

    
# References :
# https://plotly.com/python/line-charts/
# https://plotly.com/python/discrete-color/
# https://www.youtube.com/watch?v=Sb0A9i6d320&list=PLHgX2IExbFovFg4DI0_b3EWyIGk-oGRzq
# https://docs.streamlit.io/library/api-reference/layout/st.tabs
# https://www.youtube.com/watch?v=7yAw1nPareM
# https://www.youtube.com/watch?v=7yAw1nPareM
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
# https://stackoverflow.com/questions/57954510/python-plotly-bar-chart-count-items-from-csv
# https://snyk.io/advisor/python/streamlit/functions/streamlit.selectbox
# https://github.com/streamlit/streamlit/issues/949


