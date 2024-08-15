import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Custom CSS for Dark Theme
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
    }
    .main {
        background-color: #121212;
        color: #ffffff;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #1e88e5;
        border-radius: 8px;
        padding: 8px 20px;
        border: none;
        font-size: 16px;
    }
    .stTextInput>div>div>input {
        background-color: #1c1c1c;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        color: #ffffff;
        border: 1px solid #333;
    }
    .stSidebar .stSelectbox, .stSidebar .stSlider, .stSidebar .stTextInput {
        background-color: #1c1c1c;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        color: #ffffff;
        border: 1px solid #333;
    }
    .stSidebar h2 {
        color: #ffffff;
    }
    .stSidebar {
        background-color: #2c2c2c;
        padding: 20px;
    }
    .stMarkdown {
        text-align: left;
        font-size: 18px;
        color: #ffffff;
    }
    hr {
        border-color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, api_key, engine, temperature, max_tokens):
    llm = ChatOpenAI(model=engine, temperature=temperature, max_tokens=max_tokens, openai_api_key=api_key)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Title of the app
st.title("💬 Chat-Mate: Ask me anything brother...")

# Sidebar for settings
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("🔑 OpenAI API Key:", type="password")

# Select the OpenAI model
engine = st.sidebar.selectbox("🧠 OpenAI model", ["gpt-3.5-turbo"])

# Adjust response parameters
temperature = st.sidebar.slider("🌡️ Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("✍️ Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("📝 Ask any question to the chatbot below:")
user_input = st.text_input("You:")

# Button to submit the query
if st.button("Submit"):
    if user_input and api_key:
        response = generate_response(user_input, api_key, engine, temperature, max_tokens)
        st.markdown(f"**🤖 Chatbot:** {response}")
    elif user_input:
        st.warning("Please enter the OpenAI API Key in the sidebar.")
    else:
        st.write("Please provide a question to get a response.")

# Footer for additional information or branding
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #888;'>Powered by OpenAI and Langchain</div>", unsafe_allow_html=True)
