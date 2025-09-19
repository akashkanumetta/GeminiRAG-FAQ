# Install dependencies first:
# pip install langchain langchain-community langchain-google-genai faiss-cpu sentence-transformers pandas

import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# -------------------------
# 1. Load CSV
# -------------------------
csv_path = r'C:\Users\msdak\Desktop\vscode\Projects\llm\course_faqs.csv'  # your file path
df = pd.read_csv(csv_path, encoding="latin1")  # change encoding if needed

print("Columns in CSV:", df.columns)

# If your file has "Question" and "Answer" columns
if "prompt" in df.columns and "response" in df.columns:
    df["qa"] = df["prompt"] + " " + df["response"]
    loader = DataFrameLoader(df, page_content_column="qa")
else:
    # If your CSV has different columns, pick the right one
    first_col = df.columns[0]
    loader = DataFrameLoader(df, page_content_column=first_col)

documents = loader.load()

# -------------------------
# 2. Embeddings (HuggingFace, free)
# -------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------------
# 3. Vectorstore (FAISS)
# -------------------------
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# -------------------------
# 4. PaLM / Gemini LLM
# -------------------------
os.environ["GOOGLE_API_KEY"] = "your_api_key_here"  # or set in .env file
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -------------------------
# 5. Ask Questions
# -------------------------
query = "For how long this couse is valid?"
result = qa_chain.invoke({"query": query})

print("Query:", query)
print("Answer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print("-", doc.page_content[:200])
