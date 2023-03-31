import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

page_selected = st.sidebar.selectbox("Explore or Predict", ("Predict", "Explore"))
#can also add buttons/slider any widget

if page_selected == "Predict":
    show_predict_page()
elif page_selected == "Explore":
    show_explore_page()
else:
    print("Invalid Page Selected")