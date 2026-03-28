from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import streamlit as st 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="CortexDocs PDF Summarizer", layout="wide")
st.header("📄 CortexDocs PDF Summarizer")

# Initialize model
model = init_chat_model("gpt-4o-mini")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    
    # Split PDF into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
    splits = splitter.split_documents(docs)
    
    # Prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "summarize the following document"),
            ("human", "{document}")
        ]
    )
    
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser

    # Generate summary for first chunk
    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            result = chain.invoke({"document": splits[0].page_content})
            st.success("✅ Summary Generated!")
            st.text_area("Summary", value=result, height=300)