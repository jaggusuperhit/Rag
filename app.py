import streamlit as st
import time
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="RAG Demo with OpenRouter",
    page_icon="🤖",
    layout="wide"
)

# Configure OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("OpenRouter API key not found. Please add it to your .env file.")
    st.stop()

# Configure OpenRouter base URL and headers
openrouter_base_url = "https://openrouter.ai/api/v1"
openrouter_headers = {
    "HTTP-Referer": "https://localhost:8501",  # Required by OpenRouter
    "X-Title": "RAG Demo App"  # Optional, for tracking in OpenRouter dashboard
}


st.title("RAG App Demo with OpenRouter (GPT-3.5)")

st.markdown("""
This demo uses:
- OpenRouter with GPT-3.5 for the language model
- HuggingFace embeddings for document retrieval
- Streamlit for the user interface
""")

# Initialize session state for storing data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Data loading section
with st.expander("Data Source", expanded=not st.session_state.data_loaded):
    st.markdown("### Data Sources")
    st.markdown("The demo uses content from the following URLs:")
    urls = [
        'https://www.victoriaonmove.com.au/local-removalists.html',
        'https://victoriaonmove.com.au/index.html',
        'https://victoriaonmove.com.au/contact.html'
    ]
    for url in urls:
        st.markdown(f"- [{url}]({url})")

    # Set up embeddings (needed for both loading existing DB and creating new one)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Define persist directory
    persist_directory = "chroma_db"

    # Check if database already exists
    db_exists = os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0

    if not st.session_state.data_loaded:
        # Add option to use existing database if it exists
        if db_exists:
            st.info("A vector database already exists. You can use it or reload the data.")
            if st.button("Use Existing Database"):
                with st.spinner("Loading existing vector database..."):
                    try:
                        # Load existing vectorstore
                        vectorstore = Chroma(
                            persist_directory=persist_directory,
                            embedding_function=embeddings
                        )

                        # Create retriever
                        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

                        # Store in session state
                        st.session_state.retriever = retriever
                        st.session_state.data_loaded = True

                        # We don't have the original docs, but we can estimate
                        st.session_state.docs = ["Loaded from existing database"]

                        st.success("✅ Loaded existing vector database")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading existing database: {str(e)}")

        if st.button("Load Data" if not db_exists else "Reload Data"):
            with st.spinner("Loading data from URLs..."):
                # Load data
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()

                # Split data
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
                docs = text_splitter.split_documents(data)
                all_splits = docs

                # Create a persistent Chroma database

                # Check if directory exists, if not create it
                if not os.path.exists(persist_directory):
                    os.makedirs(persist_directory)

                # Create vectorstore with persistence
                try:
                    vectorstore = Chroma.from_documents(
                        documents=all_splits,
                        embedding=embeddings,
                        persist_directory=persist_directory
                    )
                    # The database is automatically persisted when using persist_directory

                    # Create retriever
                    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
                except Exception as e:
                    st.error(f"Error creating vector database: {str(e)}")
                    st.stop()

                # Store in session state
                st.session_state.retriever = retriever
                st.session_state.data_loaded = True
                st.session_state.docs = docs

                st.success(f"✅ Loaded and processed {len(docs)} document chunks")
                st.rerun()
    else:
        st.success(f"✅ Data loaded: {len(st.session_state.docs)} document chunks")

# Create LLM instance using OpenRouter with GPT-3.5
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_base=openrouter_base_url,
    openai_api_key=OPENROUTER_API_KEY,
    temperature=0.4,
    max_tokens=500,
    default_headers=openrouter_headers
)

# Get retriever from session state
if st.session_state.data_loaded:
    retriever = st.session_state.retriever
else:
    st.warning("Please load the data first")
    st.stop()



# Create a sidebar with information
with st.sidebar:
    st.header("About this Demo")
    st.markdown("""
    This is a Retrieval Augmented Generation (RAG) demo that uses:

    * **OpenRouter** with GPT-3.5 Turbo as the LLM
    * **HuggingFace** sentence-transformers for embeddings
    * **Chroma** as the vector database
    * **LangChain** for the RAG pipeline

    The app retrieves information about Victoria on Move, a moving company in Australia.
    """)

    st.markdown("### Sample Questions")
    sample_questions = [
        "What kind of services do they provide?",
        "What are their moving truck options and prices?",
        "How can I contact Victoria on Move?",
        "Do they offer interstate moving services?",
        "What is their Google rating?"
    ]

    for q in sample_questions:
        if st.button(q):
            st.session_state.query = q
            st.rerun()

# Main chat interface
st.markdown("### Ask a question about Victoria on Move")
query = st.chat_input("Type your question here...")

# Handle query from chat input or button click
if 'query' in st.session_state:
    query = st.session_state.query
    # Clear it after use
    del st.session_state.query

# Create the prompt template
system_prompt = (
    "You are an assistant for question-answering tasks about Victoria on Move, a moving company in Australia. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer based on the provided context, say that you don't know. "
    "Use three sentences maximum and keep the answer concise and professional."
    "\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


if query:
    # Display the query
    st.chat_message("user").write(query)

    with st.spinner("Generating answer..."):
        try:
            # Create the RAG chain
            question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # Get response
            response = rag_chain.invoke({"input": query})

            # Display answer in a nice format
            st.chat_message("assistant").write(response["answer"])

            # Optionally show the retrieved documents
            with st.expander("View retrieved documents"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Document {i+1}**")
                    st.markdown(f"*Source: {doc.metadata.get('source', 'Unknown')}*")
                    st.markdown(doc.page_content)
                    st.markdown("---")

            # Add a divider
            st.divider()
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

            # Provide helpful guidance based on the error
            if "no such table: collections" in str(e):
                st.warning("Database error detected. Please try reloading the data by clicking 'Reload Data' button.")
            elif "index" in str(e).lower() or "database" in str(e).lower():
                st.warning("Vector database issue detected. Try reloading the data.")

            # Add a button to clear session state and restart
            if st.button("Reset Application"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()