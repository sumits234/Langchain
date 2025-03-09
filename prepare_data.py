import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set user agent to avoid being blocked
os.environ["USER_AGENT"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/119.0.0.0 Safari/537.36")

# Load environment variables from .env
load_dotenv()

def main():
    print("Starting data preparation...")
    
    try:
        # 1. Load data from URL
        print("Loading data from website...")
        loader = WebBaseLoader("https://brainlox.com/courses/category/technical")
        docs = loader.load()
        print(f"Loaded {len(docs)} documents")

        # 2. Split documents
        print("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        print(f"Created {len(splits)} splits")

        # 3. Create embeddings using HuggingFaceEmbeddings
        print("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 4. Build the vector store
        print("Building vector store...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # 5. Save vector store locally
        print("Saving vector store...")
        vectorstore.save_local("faiss_index")
        print("Vector store saved successfully!")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
