import streamlit as st
import openai
import toml
from openai import OpenAI

st.title("Group2-CDS Batch 6: Final Project-Plant Disease Detection")

uploadedImage = st.file_uploader('Upload Leaf image for disease detection', type=['png','jpg'])



secrets = toml.load(".vscode/streamlit/secrets.toml")
client = OpenAI(api_key = secrets["OPENAI_API_KEY"])

#st.title("Chat Bot (GPT-4)")
#client.api_key = secrets["OPENAI_API_KEY"]

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
        st.markdown("Disease Name: Apple Scab" )
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