import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

load_dotenv()

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

INDEX_DIR = "faiss_index"
main_placeholder = st.empty()

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

if process_url_clicked:
    urls = [u.strip() for u in urls if u and u.strip()]
    if not urls:
        st.sidebar.warning("Please enter at least one URL.")
        st.stop()

    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    main_placeholder.text("Saving FAISS index...✅✅✅")
    vectorstore_openai.save_local(INDEX_DIR)
    time.sleep(1)
    st.sidebar.success("URLs processed and index saved.")

query = st.text_input("Question:")
if query:
    if os.path.exists(INDEX_DIR):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
        )

        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result.get("answer", ""))

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    else:
        st.warning("No index found. Please click 'Process URLs' first.")
