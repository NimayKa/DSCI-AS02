import streamlit as st 
import pandas as pd
import plotly.express as px
# must include filters 

#reference for streamlit emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.title(" 	:chart_with_upwards_trend: Additional Dashboards")
shopping_df = pd.read_csv('shopping_behavior_new_updated.csv')

# reference: https://www.youtube.com/watch?v=7yAw1nPareM
st.sidebar.header("Choose your filter:")

col1, col2 = st.columns((2.5, 3.5), gap='medium')
    
with col1:
    st.subheader("Subscription Status", divider='rainbow')
    fig_subs = px.pie(shopping_df, values="Customer ID", names="Subscription Status", color_discrete_sequence=px.colors.qualitative.Plotly, hole= 0.5)
    st.plotly_chart(fig_subs, use_container_width=True)