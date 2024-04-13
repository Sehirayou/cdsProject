import streamlit as st

st.title("Hello World")


upload= st.file_uploader('Upload Leaf image for disease detection', type=['png','jpg'])