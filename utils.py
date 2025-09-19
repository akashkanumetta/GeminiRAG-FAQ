# utils.py
# ---------------------------------------
# Utility functions for Vector DB + QA Chain
# ---------------------------------------

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()


# -------------------------
# 1. Create Vector Database
# -------------------------
def create_vector_db(csv_path: str):
    """Load CSV, create embeddings, and build FAISS vector database."""
    df = pd.read_csv(csv_path, encoding="latin1")  # adjust encoding if needed

    # If CSV has "prompt" and "response"
    if "prompt" in df.columns and "response" in df.columns:
        df["qa"] = df["prompt"] + " " + df["response"]
        loader = DataFrameLoader(df, page_content_column="qa")
    else:
        # Fallback: take first column as text
        first_col = df.columns[0]
        loader = DataFrameLoader(df, page_content_column=first_col)

    documents = loader.load()

    # HuggingFace embeddings (free & no quota issues)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAISS vector database
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


# -------------------------
# 2. Create QA Chain
# -------------------------
def create_qa_chain(vectorstore):
    """Create a RetrievalQA chain with Gemini LLM + FAISS retriever."""

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Google Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",  # fast + reliable
        temperature=0.2
    )
    
    prompt_template = """
    You are an AI assistant that answers based ONLY on the given context.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    - If the answer is in the context, give a concise and accurate response.
    - If the answer is NOT in the context, say: "I don’t know based on the provided information."
    - Do NOT make up answers.

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain
