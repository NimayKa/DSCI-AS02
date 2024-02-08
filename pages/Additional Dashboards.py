import streamlit as st 
import pandas as pd

# must include filters 

#reference for streamlit emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.title(" 	:chart_with_upwards_trend: Additional Dashboards")
shopping_df = pd.read_csv('shopping_behavior_new_updated.csv')