import streamlit as st 
import pandas as pd
import plotly.express as px
# must include filters 

#reference for streamlit emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.title(" 	:chart_with_upwards_trend: Additional Dashboards")
shopping_df = pd.read_csv('shopping_behavior_new_updated.csv')


_, cols1, cols2,  _= st.columns((0.3,4,4,0.2), gap='medium')

with cols1:
    st.markdown("                                  ")
    
    st.subheader("Subscription Status", divider='rainbow')
    st.write("Filter: Subscription Status")     
    age_group = st.selectbox('Select age group:', shopping_df['Age Group'].unique(), index=None, key='subs_selection')
    gender = st.selectbox('Select gender:', shopping_df['Gender'].unique(), index=None)
    st.markdown("-------")  
    
    if age_group:
        shopping_dff = shopping_df[shopping_df['Age Group'] == age_group]
    elif gender:
        shopping_dff = shopping_df[shopping_df['Gender'] == gender]

    colors = px.colors.qualitative.Pastel
    fig_subs = px.pie(shopping_df, values="Customer ID", names="Subscription Status", color_discrete_sequence=colors, hole= 0.5)
    fig_subs.update_traces(text = shopping_df['Subscription Status'], textposition = "outside")
    st.plotly_chart(fig_subs, use_container_width=True)
    
    st.subheader("Promo Code Used", divider='rainbow')
    st.write("Filter: Promo Code Used")   
    cust_gender = st.selectbox('Select gender:', shopping_df['Gender'].unique(), index=None, key='custgender')
    subscription = st.selectbox('Subscription Status:', shopping_df['Subscription Status'].unique(), index=None, key='sub')
    st.markdown("-------")  
    
    if cust_gender:
        promo = shopping_df[shopping_df['Gender'] == subscription].groupby('Promo Code Used').count().reset_index()
        promo = promo.rename(columns={"Customer ID": "Count"}) 
        
    elif subscription:
        promo = shopping_df[shopping_df['Subscription Status'] == subscription].groupby('Promo Code Used').count().reset_index()
        promo = promo.rename(columns={"Customer ID": "Count"}) 
        
    else:
        promo = shopping_df.groupby('Promo Code Used').count().reset_index()
        promo = promo.rename(columns={"Customer ID": "Count"})
    
    colors = px.colors.qualitative.Pastel
    fig_promo = px.pie(promo, values="Count", names="Promo Code Used", color_discrete_sequence=colors, hole= 0.5)
    st.plotly_chart(fig_promo, use_container_width=True)

with cols2:
    st.markdown("                                  ")
    
    st.subheader("Payment Method Preferred", divider='rainbow')
    st.write("Filter: Payment Method Prefered")   
    options_by_gender = st.selectbox('Select gender:', shopping_df['Gender'].unique(), index=None, key='selectionbygender')
    category = st.selectbox('Select Category', shopping_df['Category'].unique(), index=None, key='categoryselection')
    st.markdown("-------")  
    
    
    if options_by_gender:
            payment = shopping_df[shopping_df['Gender'] == options_by_gender].groupby('Payment Method').count().reset_index()
            payment = payment.rename(columns={"Customer ID": "Count"})
            payment.sort_values(by='Count', ascending=False, inplace=True)
    elif category:
        payment = shopping_df[shopping_df['Category'] == category].groupby('Payment Method').count().reset_index()
        payment = payment.rename(columns={"Customer ID": "Count"})
        payment.sort_values(by='Count', ascending=False, inplace=True)
    else:
        payment = shopping_df.groupby('Payment Method').count().reset_index()
        payment = payment.rename(columns={"Customer ID": "Count"})
    
    colors = px.colors.qualitative.Pastel
    fig_payment = px.bar(
                payment,
                x="Payment Method",
                y="Count",
                orientation="v",
                color="Payment Method",
                color_discrete_sequence=colors,
                template="plotly",
                # labels={'x':'Payment Method', 'y': 'Counts'}
                # text= shopping_df['Customer ID'].apply(lambda x: f'{x/1000:.1f}')
                )
    st.plotly_chart(fig_payment, use_container_width=True)
    
    st.subheader("Popular Clothing Sizes", divider='rainbow')
    st.write("Filter: Popular Clothing Sizes")   
    sales_bysizes = st.checkbox('Total Sales by Clothing Sizes')
    gender = st.selectbox('Select gender:', shopping_df['Gender'].unique(), index=None, key='selectiongender')
    st.markdown("-------")  
    
    if sales_bysizes:
        clothing = shopping_df.groupby('Size')['Purchase Amount (USD)'].sum().reset_index()
        clothing.sort_values(by='Purchase Amount (USD)', ascending=False, inplace=True)
            
    elif gender:
            clothing = shopping_df[shopping_df['Gender'] == gender].groupby('Size').count().reset_index()
            clothing = clothing.rename(columns={"Customer ID": "Count"})
            clothing.sort_values(by='Count', ascending=False, inplace=True)
    else:
        clothing = shopping_df.groupby("Size").count().reset_index()
        clothing = clothing.rename(columns={"Customer ID": "Count"})
    
    colors = px.colors.qualitative.Pastel
    if sales_bysizes:
        fig_clothing = px.bar(
                    clothing,
                    x="Purchase Amount (USD)",
                    y="Size",
                    orientation="h",
                    color="Size",
                    color_discrete_sequence=colors,
                    template="presentation"
                    )
    else:
        fig_clothing = px.bar(
                clothing,
                x="Count",
                y="Size",
                orientation="h",
                color="Size",
                color_discrete_sequence=colors,
                template="presentation"
                )
    
    st.plotly_chart(fig_clothing, use_container_width=True)

_, col1, col2,  _= st.columns((0.3,4,4,0.2), gap='medium')
with col1:
    st.subheader("Customer Age", divider='rainbow')
    st.write("Filter: Customer Age")   
    totalsales = st.checkbox('Total Sales by Age Group')
    gender_customer = st.selectbox('Select gender:', shopping_df['Gender'].unique(), index=None, key='selectioncustgender')
    st.markdown("-------") 
    
    if totalsales:
        age = shopping_df.groupby('Age Group')['Purchase Amount (USD)'].sum().reset_index()
        age.sort_values(by='Purchase Amount (USD)', ascending=False, inplace=True)
            
    elif gender_customer:
        age = shopping_df[shopping_df['Gender'] == gender_customer].groupby('Age Group').count().reset_index()
        age = age.rename(columns={"Customer ID": "Count"})
    else:
        age = shopping_df.groupby('Age Group').count().reset_index()
        age = age.rename(columns={"Customer ID": "Count"})
    
    
    colors = px.colors.qualitative.Pastel
    if totalsales:
         fig_age = px.bar(
            age,
            x="Age Group",
            y="Purchase Amount (USD)",
            orientation="v",
            color="Age Group",
            color_discrete_sequence=colors,
            template="presentation"
            
    )
    else:
        fig_age = px.bar(
                age,
                x="Age Group",
                y="Count",
                orientation="v",
                color="Age Group",
                color_discrete_sequence=colors,
                template="presentation"
                
        )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    st.subheader("Popular Color", divider='rainbow')
    st.write("Filter: Popular Color")   
    sales = st.checkbox('Total Sales by Color')
    options_top_color = st.selectbox('Select Top Colors:', ['All Items', 'Top 5 Colors'])
    st.markdown("-------")  
    
    if sales:
        color = shopping_df.groupby('Color')['Purchase Amount (USD)'].sum().reset_index()
        color.sort_values(by='Purchase Amount (USD)', ascending=False, inplace=True)
            
    elif options_top_color == 'Top 5 Colors':
        color = shopping_df.groupby('Color').count().reset_index().head(5)
        color = color.rename(columns={"Customer ID": "Count"})
        color.sort_values(by='Count', ascending=False, inplace=True)
    else:
        color = shopping_df.groupby("Color").count().reset_index()
        color = color.rename(columns={"Customer ID": "Count"})
    
    
    colors = px.colors.qualitative.Pastel
    if sales:
        fig_clothing = px.bar(
                color,
                x="Purchase Amount (USD)",
                y="Color",
                orientation="h",
                color="Color",
                color_discrete_sequence=colors,
                template="presentation"
                )
    else:
        fig_clothing = px.bar(
                    color,
                    x="Count",
                    y="Color",
                    orientation="h",
                    color="Color",
                    color_discrete_sequence=colors,
                    template="presentation"
                    )
        
    st.plotly_chart(fig_clothing, use_container_width=True)


# reference: https://www.youtube.com/watch?v=7yAw1nPareM
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


