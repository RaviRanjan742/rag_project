import sys
import platform
if platform.system() != "Windows":
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        print("Warning: pysqlite3 not installed. Using system SQLite.")
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import pickle
from dotenv import load_dotenv
from typing import List, Set
import shutil
import hashlib
import time

# Load environment variables from .env file
load_dotenv()

# Initialize SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Chroma client
client = chromadb.PersistentClient(path="./chroma_db", settings=Settings())

# Initialize Groq client with error handling
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    groq_client = None

# File to store processed files list
PROCESSED_FILES_PATH = "data/processed_files.pkl"

def create_embeddings(chunks: List[str], filename: str) -> None:
    """Create embeddings for chunks and store in a unique Chroma collection per file."""
    try:
        # Create a unique collection name using filename and timestamp
        timestamp = str(time.time()).replace(".", "_")
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        collection_name = f"{file_hash}_{timestamp}"
        collection = client.get_or_create_collection(name=collection_name)
        
        # Generate embeddings
        embeddings = model.encode(chunks, show_progress_bar=False)
        
        # Store in Chroma with metadata
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=[f"{filename}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"filename": filename} for _ in chunks]
        )
        print(f"Embeddings created for {filename} in collection {collection_name}")
    except Exception as e:
        print(f"Error creating embeddings for {filename}: {e}")

def query_vector_db(query: str, n_results: int = 3) -> str:
    """Query all Chroma collections and summarize results with reduced input size."""
    if not groq_client:
        return "Error: Groq client not initialized. Please set GROQ_API_KEY in .env file."
    
    try:
        # Get all collections
        collections = client.list_collections()
        if not collections:
            return "No documents in any collection. Please upload documents first."
        
        # Generate query embedding
        query_embedding = model.encode([query])[0].tolist()
        
        # Query all collections and aggregate results with distances
        all_results = []
        for collection in collections:
            query_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            if query_results["documents"]:
                for doc, meta, dist in zip(
                    query_results["documents"][0],
                    query_results["metadatas"][0],
                    query_results["distances"][0]
                ):
                    # Truncate document to 500 characters to reduce token usage
                    truncated_doc = doc[:500] + "..." if len(doc) > 500 else doc
                    all_results.append((truncated_doc, meta, dist))
        
        if not all_results:
            return "No relevant documents found for the query."
        
        # Sort by distance (lower is better) and take top n_results
        all_results.sort(key=lambda x: x[2])  # Sort by distance
        top_results = all_results[:n_results]
        documents = [result[0] for result in top_results]
        metadatas = [result[1] for result in top_results]
        
        # Combine documents for summarization with source information
        combined_text = ""
        for doc, meta in zip(documents, metadatas):
            source = meta.get("filename", "Unknown") if meta else "Unknown"
            combined_text += f"Source: {source}\nContent: {doc}\n\n"
        
        # Optimized prompt to reduce token usage
        prompt = (
            "Summarize the following in concise bullet points, focusing on key points for '{}':\n\n{}"
        ).format(query, combined_text)
        
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Provide concise bullet-point summaries with source attribution."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        print(f"Error querying vector DB or summarizing: {e}")
        return f"Error occurred while processing query: {str(e)}"

def load_processed_files() -> Set[str]:
    """Load the set of processed files from disk."""
    try:
        if os.path.exists(PROCESSED_FILES_PATH):
            with open(PROCESSED_FILES_PATH, 'rb') as f:
                return pickle.load(f)
        return set()
    except Exception as e:
        print(f"Error loading processed files: {e}")
        return set()

def save_processed_files(processed_files: Set[str]) -> None:
    """Save the set of processed files to disk."""
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(PROCESSED_FILES_PATH), exist_ok=True)
        with open(PROCESSED_FILES_PATH, 'wb') as f:
            pickle.dump(processed_files, f)
    except Exception as e:
        print(f"Error saving processed files: {e}")

def clear_all_data() -> None:
    """Clear all stored embeddings and processed files data."""
    try:
        # Delete the Chroma database directory
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        
        # Delete processed files list
        if os.path.exists(PROCESSED_FILES_PATH):
            os.remove(PROCESSED_FILES_PATH)
        
        # Delete chunks directory
        if os.path.exists("chunks"):
            shutil.rmtree("chunks")
            os.makedirs("chunks")
        
        print("All data cleared successfully!")
    except Exception as e:
        print(f"Error clearing data: {e}")

def get_collection_stats() -> dict:
    """Get statistics about the stored documents."""
    try:
        collections = client.list_collections()
        total_chunks = 0
        unique_files = set()
        
        for collection in collections:
            count = collection.count()
            total_chunks += count
            # Get one item to extract filename
            if count > 0:
                items = collection.get()
                metadatas = items["metadatas"]
                if metadatas:
                    unique_files.add(metadatas[0]["filename"])
        
        return {
            "total_chunks": total_chunks,
            "unique_files": len(unique_files),
            "file_names": list(unique_files)
        }
    except Exception as e:
        print(f"Error getting collection stats: {e}")
        return {"total_chunks": 0, "unique_files": 0, "file_names": []}