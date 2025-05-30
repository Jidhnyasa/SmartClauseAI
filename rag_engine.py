from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_TOKEN")

llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct", huggingfacehub_api_token=hf_token,
    model_kwargs={"temperature": 0.8, "max_new_tokens": 512},
)

def ask_question(clause_dict, query):
    text_chunks = []
    for label, text in clause_dict.items():
        text_chunks.append(f"{label}: {text}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text("\n".join(text_chunks))]

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embedding)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    return qa_chain.run(query)

__all__ = ["ask_question"]