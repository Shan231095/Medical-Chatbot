import streamlit as st
from streamlit_chat import message
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="GEMINI_API_KEY")

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Medical system prompt
medical_prompt = (
    "You are a helpful and knowledgeable medical assistant chatbot. "
    "Provide accurate, clear, and concise medical information based on user questions. "
    "Always remind users to consult a healthcare professional for diagnosis and treatment."
)

# Function to ask the chatbot
def ask_medical_chatbot(history, user_input):
    conversation = "\n".join(history + [f"User: {user_input}", "Assistant:"])
    prompt = f"{medical_prompt}\n{conversation}"
    response = model.generate_content(prompt)
    return response.text

# Streamlit app
st.set_page_config(page_title="Medical Chatbot", page_icon=":hospital:", layout="centered")
st.title("Medical Chatbot - Powered by Gemini")
st.info("Disclaimer: This chatbot provides general medical information and is not a substitute for professional medical advice.")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
user_input = st.text_input("Ask a medical question:")

if st.button("Send") and user_input:
    reply = ask_medical_chatbot(st.session_state.history, user_input)
    st.session_state.history.append(f"User: {user_input}")
    st.session_state.history.append(f"Assistant: {reply}")
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": reply})

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        message(msg["content"], is_user=True)
    else:
        message(msg["content"])

# Clear chat option
if st.button("Clear Chat"):
    st.session_state.history = []
    st.session_state.messages = []