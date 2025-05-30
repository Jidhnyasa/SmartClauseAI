import streamlit as st
from clause_classifier import classify_clauses
from rag_engine import ask_question

st.set_page_config(page_title="SmartClause AI", layout="wide")
st.title("📜 SmartClause AI - Contract Intelligence Assistant")

uploaded_file = st.file_uploader("Upload a Contract (PDF or DOCX)", type=["pdf", "docx"])
question = st.text_input("Ask a question about the contract")

if uploaded_file:
    with st.spinner("Classifying clauses..."):
        clause_texts = classify_clauses(uploaded_file)
        st.subheader("📑 Extracted Clauses")
        for label, text in clause_texts.items():
            st.markdown(f"### {label}\n{text[:500]}...\n")

    if question:
        with st.spinner("Answering using LLaMA..."):
            answer = ask_question(clause_texts, question)
            st.success(f"Answer: {answer}")