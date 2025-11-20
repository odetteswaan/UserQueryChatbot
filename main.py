import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


pinecone_api_key = os.getenv("PINECONE_API_KEY")
hugging_face_api_key = os.getenv("HF_API_KEY")

# Pinecone index
IndexName = "chatbot"

# Streamlit Page
st.set_page_config(page_title="Conversational Finance Chatbot", layout="wide")
st.title("🅰️irtel Finance Conversational Chatbot")

# Initialize session messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input UI
customer_id = st.text_input("Enter Customer ID")
user_query = st.chat_input("Ask a question…")

# Setup Pinecone + LLM
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(IndexName)

model = SentenceTransformer("BAAI/bge-large-en")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=hugging_face_api_key,
    task="text-generation",
    temperature=0.2,
    max_new_tokens=250
)

chat_model = ChatHuggingFace(llm=llm)

# Show Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# When user sends a message:
if user_query:
    if customer_id.strip() == "":
        st.error("Please enter Customer ID first.")
        st.stop()

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display user message
    with st.chat_message("user"):
        st.write(user_query)

    # ----- Vector Query -----
    combined_query = f"{user_query} (customer id: {customer_id})"
    query_emb = model.encode(combined_query).tolist()

    response = index.query(
        vector=query_emb,
        top_k=5,
        include_metadata=True
    )

    try:
        context = response.matches[0].metadata["text"]
    except:
        context = "No customer data found."

    # ----- Prompt -----
    prompt = f"""
You are an Airtel Finance AI Assistant.
Answer based on customer details and chat history.

Chat History:
{[m for m in st.session_state.messages]}

Customer ID: {customer_id}

Customer Data:
{context}

User Query:
{user_query}

Respond in simple, clear English.
"""

    # ----- LLM Response -----
    bot_reply = chat_model.invoke(prompt).content

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # Display response
    with st.chat_message("assistant"):
        st.write(bot_reply)
