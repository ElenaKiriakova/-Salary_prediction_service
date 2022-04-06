import streamlit as st
from predict_page import show_predict_page
from dash0 import dash0

page = st.sidebar.selectbox('Menu', ('Predictions', 'Statistics'))

if page == 'Predictions':
    show_predict_page()
else:
    dash0()



