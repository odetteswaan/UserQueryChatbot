import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone , ServerlessSpec
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
IndexName='chatbot'
pinecone_api_key=os.getenv("PINECONE_API_KEY")
hugging_face_api_key=os.getenv("HF_API_KEY")
st.title("Customer Query Tool")

# Input fields
customer_id = st.text_input("Enter Customer ID")
user_query = st.text_area("Enter your Query")
pc=Pinecone(api_key=pinecone_api_key)
index=pc.Index(IndexName)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=hugging_face_api_key,
    task="text-generation",
    temperature=0,
    max_new_tokens=200
)
model = SentenceTransformer("BAAI/bge-large-en")
chat_model = ChatHuggingFace(llm=llm)
# Button
if st.button("Submit"):
    if customer_id.strip() == "" or user_query.strip() == "":
        st.error("Please fill both fields.")
    else:
        # Combine both in f-string
        combined_query = f"{user_query} my customer id is {customer_id}"
        query_emb = model.encode(combined_query).tolist()
        response = index.query(
            vector=query_emb,
            top_k=5,
            include_metadata=True
        )
        context = response.matches[0].metadata['text']

        prompt = f"""
        You are a finance assistant for Airtel.
        Use the following customer data to answer the question.

        Customer Data:
        {context}

        User Question:
        {combined_query}

        Answer in simple and clear English.
        """

        st.subheader("Combined Query:")
        st.write(combined_query)

        # Dummy response logic (you can replace with your vector search)
        response = f"I have processed your query for customer {customer_id}. You asked: '{user_query}'."

        st.subheader("Response:")
        st.write(chat_model.invoke(prompt).content)
