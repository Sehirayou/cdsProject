import streamlit as st

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.models import load_model

import numpy as np
import time

import openai
import toml
from openai import OpenAI


tab1, tab2, tab3 = st.tabs(["Home", "Solution", "Team"])

#First Tab: Title of Application and description
with tab1:
    st.title("Plant Disease Detection App")
    st.markdown("Crop losses due to diseases pose a significant threat to food security worldwide. Traditional methods of disease detection are often labor-intensive and time-consuming, leading to delayed diagnosis and ineffective treatments. Leveraging deep learning techniques, this app aims to help farmers detect plant diseases and suggest remidial measures, by uploading images of the infected leaves")



#Second Tab: Image upload and disease detection and remidy susgestions
with tab2: 
    st.title("Upload leaf Image for plant classification,disease detection and Remediation methods")
    uploadedImage = st.file_uploader('Upload Leaf image', type=['png','jpg'])

    secrets = toml.load(".vscode/streamlit/secrets.toml")
    client = OpenAI(api_key = secrets["OPENAI_API_KEY"])
    
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #if prompt := st.chat_input("What is up?"):
    if uploadedImage is not None:
        prompt = "Remidy for Apple Scab"
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            #disease_details = st.empty()
            st.markdown("Disease Name: Apple Scab" )
            st.markdown("Severity of Disease: Mild" )
            st.markdown("Searching for " + prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                #full_response += response.choices[0].delta.get("content", "")
                full_response += response.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})





# Third Tab
with tab3:
    st.title("CDS Batch 6 - Group 2:")
    st.divider()

    st.write("Abhinav Singh")
    st.divider()  
    
    st.write("Ankit Kourav")
    st.divider()  

    st.write("Challoju Anurag.")
    st.divider()  

    st.write("Madhucchand Darbha")
    st.divider()  

    st.write("Neha Gupta")
    st.divider()  

    st.write("Pradeep Rajagopal")
    st.divider()  

    st.write("Rakesh Vegesana")
    st.divider()  

    st.write("Sachin Sharma")
    st.divider()  

    st.write("Shashank Srivastava")
    st.divider()      