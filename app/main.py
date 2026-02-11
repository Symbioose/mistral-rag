import streamlit as st
import os
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Mistral RAG")
st.title("Mistral RAG : Pose tes questions a tes documents")
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Mistral API Key", type="password")


@st.cache_resource
def get_vector_store():
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "mistral_collection")
    client = QdrantClient(url=url)

    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

    embedding_size = len(embeddings.embed_query("dimension check"))
    if not client.collection_exists(collection_name):
         client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vector_store

def process_document(uploaded_file):
    file_path = f"./documents/{uploaded_file.name}"
    os.makedirs("./documents", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    
    return len(chunks)


if not api_key:
    st.warning("Entre ta cle API Mistral pour commencer.")
    st.stop()

try:
    get_vector_store()
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("📂 Tes Documents")
    uploaded_file = st.file_uploader("Upload un PDF ou TXT", type=["pdf", "txt"])
    if uploaded_file and st.button("Indexer le document"):
        with st.spinner("Indexation en cours..."):
            num_chunks = process_document(uploaded_file)
            st.success(f"Document indexe en {num_chunks} morceaux !")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pose ta question sur tes documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        llm = ChatMistralAI(
            mistral_api_key=api_key, 
            model="mistral-large-latest", 
            temperature=0.2
        )

        template = """Tu es un assistant IA expert et sympa.
        Utilise les elements de contexte suivants pour repondre a la question de l'utilisateur.
        Si tu ne connais pas la reponse, dit le clairement, soit honnete, n'invente rien.
        Sois concis.
        
        Contexte : {context}
        Question : {question}
        
        Reponse :"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        response = qa_chain.invoke({"query": prompt})
        answer = response["result"]
        
        message_placeholder.markdown(answer)
        
        with st.expander("Voir les sources utilisees"):
            for doc in response["source_documents"]:
                st.caption(doc.page_content[:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
