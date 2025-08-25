# Run : https://gpu-instance-r8zy.notebook.us-west-2.sagemaker.aws/proxy/8501/

import streamlit as st
from response import chatbot
import time
import re

def stream_message(message):
    for token in message:
        yield token
        time.sleep(0.02)

with st.sidebar:
    st.write("All goes here")

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    result = chatbot.process_query(prompt)
    msg = result['response']
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write_stream(stream_message(msg))
    