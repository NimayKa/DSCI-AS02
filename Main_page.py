import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title = "Consumer Behavior and Shopping Habits in United States",
    layout="wide"
    
)

st.title(":bar_chart: Consumer Behavior and Shopping Habits in United States")
# for white space
for _ in range (3):
    st.markdown("") 
st.sidebar.success("Select a page.") 

shopping_df = pd.read_csv('shopping_behavior_new_updated.csv')


#Top KPI's
total_sales = int(shopping_df['Purchase Amount (USD)'].sum())
average_rating = round(shopping_df['Review Rating'].mean(), 1)
star_rating = ":star:" * int(round(average_rating, 0))
average_age = round(shopping_df['Age'].mean())

_, col1, col2, col3, _ = st.columns((0.3, 2.5, 2.5, 2.5, 0.3), gap='medium')

with col1:
    st.subheader("Total Sales:")
    st.subheader(f"US $ {total_sales:,}")

with col2:
    st.subheader("Average Rating:")
    st.subheader(f"{average_rating} {star_rating}")  
    
with col3:
    st.subheader("Average Age:")
    st.subheader(f"{average_age} years")

# for white space
for _ in range (3):
    st.markdown("")  

st.markdown("---------")

_, cols1, cols2, cols3, _= st.columns((0.3,2.5,4,7,0.2), gap='medium')
with cols1:
    st.markdown("                                  ")
    st.write("Filter: Customer's Gender Chart")
    all_age = st.checkbox('For All age')
    slider_age = st.slider('Age slider', min_value=shopping_df['Age'].min(), max_value=shopping_df['Age'].max(), key='sliderage')
            
    options_by_fop = st.selectbox('Select frequency of purchases:', shopping_df['Frequency of Purchases'].unique(), index=None )
    st.markdown("-------") 
    st.write("Filter: Subscription in United States")
    subscription = st.radio('Subscription Status:', ['Yes', 'No'])
    
            
with cols2:
    st.subheader("Gender", divider='rainbow')
            
    if all_age:
        filtered =  shopping_df
    elif slider_age <= 19:
        st.write("Age Group: Teenager")
        filtered = shopping_df[shopping_df['Age'] == slider_age]
    elif slider_age >= 20 and slider_age <= 24:
        st.write("Age Group: Young Adult")
        filtered = shopping_df[shopping_df['Age'] == slider_age]
    elif slider_age >= 25 and slider_age <= 49:
        st.write("Age Group: Adult")
        filtered = shopping_df[shopping_df['Age'] == slider_age]
    else:
        st.write("Age Group: Old")
        filtered = shopping_df[shopping_df['Age'] == slider_age]
                
    if options_by_fop:    
        filtered = shopping_df[shopping_df['Frequency of Purchases'] == options_by_fop]
            
    colors = px.colors.qualitative.Pastel
            
    fig_gender = px.pie(filtered, values="Purchase Amount (USD)", names="Gender", color_discrete_sequence=colors, hole= 0.5, template="plotly")
    fig_gender.update_traces(text = filtered['Gender'], textposition = "outside")
                    
    st.plotly_chart(fig_gender, use_container_width=True)
    
with cols3:
    st.subheader("Subscription in United States", divider='rainbow')
    loc = pd.read_csv('us_states.csv')
        
    # merge dataframe
    new_df = pd.merge(shopping_df, loc, left_on='Location', right_on='us states', how='left')

    if subscription:
        top = new_df[new_df['Subscription Status'] == subscription].groupby('Abbreviation').count().reset_index()
        top = top.rename(columns={"Customer ID": "Count"}) 
        top.sort_values(by='Count', ascending=False, inplace=True)
        
    if subscription == 'Yes':
        fig = px.choropleth(
                            top,
                            locations='Abbreviation',
                            color='Count',
                            color_continuous_scale='tealrose_r',
                            hover_name='Location',
                            locationmode='USA-states',
                            scope='usa',
                            labels={'Count':'Number of Subscription'}
        )
    else:
        fig = px.choropleth(
                            top,
                            locations='Abbreviation',
                            color='Count',
                            color_continuous_scale='tealrose_r',
                            hover_name='Location',
                            locationmode='USA-states',
                            scope='usa',
                            labels={'Count':'Number of Non-Subscription'}
        )
        
            
    fig.add_scattergeo(
                locations=new_df['Abbreviation'],
                locationmode='USA-states',
                text=new_df['Abbreviation'],
                mode='text'
                )
                
    st.plotly_chart(fig, use_container_width=True)
    
   
    # for white space
    for _ in range (3):
        st.markdown("")
        
                
    # _ for empty column         
_, col1s, col2s, col3s, _ = st.columns((0.3,3,5,7,0.2), gap='medium')
            
with col1s:
    for _ in range (2):
        st.markdown("")
    st.markdown("Filter: Popular Product Category chart")
    sales_by_category = st.checkbox('Total Sales by Product Category')
    options_by_age = st.selectbox('Select age group:', shopping_df['Age Group'].unique(), index=None)
    st.markdown("------")
    st.markdown("Filter: Top Item Purchased chart")
    sales_by_item = st.checkbox('Total Sales by Items')
    options_top_item = st.selectbox('Select Top Items:', ['All Items', 'Top 5 Item Purchased'])
    options_by_gender = st.selectbox('Select gender:', shopping_df['Gender'].unique(), index=None)
        # options_by_subscription = st.selectbox('Select Subscription Status:', shopping_df['Subscription Status'].unique(), index=None)
                
with col2s:
    st.markdown("                                  ")
    st.markdown("                                  ")
    st.subheader("Popular Product Category", divider='rainbow')
            
    if sales_by_category:
        popular = shopping_df.groupby('Category')['Purchase Amount (USD)'].sum().reset_index()
        popular.sort_values(by='Purchase Amount (USD)', ascending=False, inplace=True)
            
    elif options_by_age:
        popular = shopping_df[shopping_df['Age Group'] == options_by_age].groupby('Category').count().reset_index()
        popular = popular.rename(columns={"Customer ID": "Count"})
        popular.sort_values(by='Count', ascending=False, inplace=True)
    else:
        popular = shopping_df.groupby('Category').count().reset_index()
        popular = popular.rename(columns={"Customer ID": "Count"})
        popular.sort_values(by='Count', ascending=True, inplace=True)


    colors = px.colors.qualitative.Pastel
    if sales_by_category:        
        fig_category = px.bar(
                            popular,
                            x="Category",
                            y="Purchase Amount (USD)",
                            orientation="v",
                            color="Category",
                            color_discrete_sequence=colors,
                            template="plotly",
                            )
    else:
        fig_category = px.bar(
                            popular,
                            x="Category",
                            y="Count",
                            orientation="v",
                            color="Category",
                            color_discrete_sequence=colors,
                            template="plotly",
                            )
                    
                    
    st.plotly_chart(fig_category, use_container_width=True)
        
                
                
with col3s:
    st.markdown("                                  ")
    st.markdown("                                  ")
    st.subheader("Popular Item Purchased", divider='rainbow')
            
    if sales_by_item:
        popular_item = shopping_df.groupby('Item Purchased')['Purchase Amount (USD)'].sum().reset_index()
        popular_item.sort_values(by='Purchase Amount (USD)', ascending=False, inplace=True)

    elif options_top_item == 'Top 5 Item Purchased':
        popular_item = shopping_df.groupby('Item Purchased').count().reset_index().head(5)
        popular_item = popular_item.rename(columns={"Customer ID": "Count"})
        popular_item.sort_values(by='Count', ascending=False, inplace=True)
    else:
        popular_item = shopping_df.groupby('Item Purchased').count().reset_index()
        popular_item = popular_item.rename(columns={"Customer ID": "Count"})
        popular_item.sort_values(by='Count', ascending=False, inplace=True)
        
    if options_by_gender:
            popular_item = shopping_df[shopping_df['Gender'] == options_by_gender].groupby('Item Purchased').count().reset_index().head(5)
            popular_item = popular_item.rename(columns={"Customer ID": "Count"})
            popular_item.sort_values(by='Count', ascending=False, inplace=True)

    colors = px.colors.qualitative.Pastel
        
    if sales_by_item:       
            fig_item = px.bar(
                            popular_item,
                            x="Purchase Amount (USD)",
                            y="Item Purchased",
                            orientation="h",
                            color="Item Purchased",
                            color_discrete_sequence=colors,
                            template="plotly",
                            text= popular_item['Purchase Amount (USD)'].apply(lambda x: f'{x/1}')
                            )
    else:
        fig_item = px.bar(
                            popular_item,
                            x="Count",
                            y="Item Purchased",
                            orientation="h",
                            color="Item Purchased",
                            color_discrete_sequence=colors,
                            template="plotly",
                            text= popular_item['Count'].apply(lambda x: f'{x/1}')
                            )
    st.plotly_chart(fig_item, use_container_width=True)
                
_, row1, _, row2, _ = st.columns((0.3, 5,0.3,5, 0.3))
with row1:
    for _ in range (5):
        st.markdown("") 
         
    st.subheader("Popular Shipping Type", divider='rainbow')    
    options_seasons = st.selectbox('Select season:', shopping_df['Season'].unique(), index=None)
    sales_by_shipping = st.checkbox('Total Sales by Shipping Type')
    
    if options_seasons:
        shipmode = shopping_df[shopping_df['Season'] == options_seasons].groupby('Shipping Type').count().reset_index()
        shipmode = shipmode.rename(columns={"Customer ID": "Count"})
        shipmode.sort_values(by='Count', ascending=True, inplace=True)
    
    elif sales_by_shipping:
        shipmode = shopping_df.groupby('Shipping Type')['Purchase Amount (USD)'].sum().reset_index()
        shipmode.sort_values(by='Purchase Amount (USD)', ascending=False, inplace=True)
    else:
        shipmode = shopping_df.groupby('Shipping Type').count().reset_index()
        shipmode = shipmode.rename(columns={"Customer ID": "Count"})
        shipmode.sort_values(by='Count', ascending=True, inplace=True)

                
    colors = px.colors.qualitative.Pastel            
    
    if sales_by_shipping:
        fig_shipping = px.bar(
                                shipmode,
                                    x="Shipping Type",
                                    y="Purchase Amount (USD)",
                                    orientation="v",
                                    color="Shipping Type",
                                    color_discrete_sequence=colors,
                                    template="plotly_white",
                                    text= shipmode['Purchase Amount (USD)'].apply(lambda x: f'{x/1}')
                                    )
    else:
        fig_shipping = px.bar(
                            shipmode,
                                x="Shipping Type",
                                y="Count",
                                orientation="v",
                                color="Shipping Type",
                                color_discrete_sequence=colors,
                                template="plotly_white",
                                text= shipmode['Count'].apply(lambda x: f'{x/1}')
                                )
    st.plotly_chart(fig_shipping, use_container_width=True)             
    
    

with row2:
        # For white space
    for _ in range (5):
        st.markdown("") 
        
    st.subheader("Total Sales by Seasons", divider='rainbow')
    category = st.selectbox('Select Category', shopping_df['Category'].unique(), index=None)
    
    if category:
        season = shopping_df[shopping_df['Category'] == category].groupby('Season')['Purchase Amount (USD)'].sum().sort_values(ascending=False)
    else:                                
        season = shopping_df.groupby('Season')['Purchase Amount (USD)'].sum().sort_values(ascending=False)
                        
    fig_season = px.line(season,
                                x=season.index,
                                y='Purchase Amount (USD)',
                                template="plotly_white",
                                color_discrete_sequence=colors,
                                height=500,
                                markers=True)
                            
    st.plotly_chart(fig_season, use_container_width=True)
            
    
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
    # https://plotly.com/python/setting-graph-size/
    # https://www.digitalocean.com/community/tutorials/pandas-merge-two-dataframe


