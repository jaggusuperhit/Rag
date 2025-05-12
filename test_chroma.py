import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Define persist directory
persist_directory = "chroma_db"

# Create directory if it doesn't exist
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Set up embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Create some test documents
documents = [
    Document(page_content="Victoria on Move is a moving company in Australia.", metadata={"source": "test"}),
    Document(page_content="You can contact Victoria on Move at info@victoriaonmove.com.au.", metadata={"source": "test"}),
    Document(page_content="Victoria on Move offers local and interstate moving services.", metadata={"source": "test"})
]

# Create and persist the database
try:
    print("Creating vector database...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # The database is automatically persisted when using persist_directory
    print("Database created and persisted successfully!")

    # Test a query
    print("\nTesting query...")
    results = vectorstore.similarity_search("How can I contact Victoria on Move?", k=1)
    print(f"Query result: {results[0].page_content}")

    print("\nTest completed successfully!")
except Exception as e:
    print(f"Error: {str(e)}")
