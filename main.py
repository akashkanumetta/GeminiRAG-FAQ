import streamlit as st
from utils import create_vector_db, create_qa_chain

CSV_PATH = r"C:\Users\msdak\Desktop\vscode\Projects\llm\course_faqs.csv"

@st.cache_resource
def load_chain():
    vectorstore = create_vector_db(CSV_PATH)
    qa_chain = create_qa_chain(vectorstore)
    return qa_chain

qa_chain = load_chain()

st.title("📚 FAQ Assistant")
user_query = st.text_input("Ask me a question:")

if user_query:
    result = qa_chain.invoke({"query": user_query})
    st.subheader("Answer:")
    st.write(result["result"])
