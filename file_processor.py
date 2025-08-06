import PyPDF2
import pandas as pd
import json
import os
from typing import List
import re

def process_file(file) -> List[str]:
    """Process different file types and return paragraph-based text chunks."""
    try:
        file_extension = os.path.splitext(file.name)[1].lower()
        text = ""
        
        if file_extension == ".txt":
            text = file.read().decode("utf-8", errors="ignore")
        elif file_extension == ".pdf":
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                text += extracted or ""
        elif file_extension == ".csv":
            df = pd.read_csv(file)
            # Convert CSV to a more readable text format
            text = f"Dataset: {file.name}\n\n"
            text += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            text += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
            
            # Add summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                text += "Numeric Column Statistics:\n"
                for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                    text += f"- {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}\n"
                text += "\n"
            
            # Add sample data
            text += "Sample Data:\n"
            text += df.head(10).to_string(index=False)
            
        elif file_extension == ".json":
            data = json.load(file)
            text = json.dumps(data, indent=2)
        else:
            return []
        
        # Clean and normalize text
        text = clean_text(text)
        
        # Smart chunking based on content type
        chunks = smart_chunk_text(text, file_extension)
        
        return chunks if chunks else [text[:1000]]  # Fallback to first 1000 chars
    except Exception as e:
        print(f"Error processing file {file.name}: {e}")
        return []

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s\.,;:!?()-]', ' ', text)
    
    return text.strip()

def smart_chunk_text(text: str, file_extension: str) -> List[str]:
    """Intelligently chunk text based on content type and structure."""
    chunks = []
    
    if file_extension == ".json":
        # For JSON, try to chunk by logical sections
        try:
            # Split by major JSON structure elements
            sections = text.split('\n  "')
            current_chunk = ""
            
            for section in sections:
                if len(current_chunk + section) > 800:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = section
                else:
                    current_chunk += "\n  \"" + section if current_chunk else section
            
            if current_chunk:
                chunks.append(current_chunk.strip())
                
        except:
            # Fallback to paragraph chunking
            chunks = paragraph_chunk(text)
    
    elif file_extension == ".csv":
        # For CSV, chunk by logical sections (header, stats, data)
        sections = text.split('\n\n')
        current_chunk = ""
        
        for section in sections:
            if len(current_chunk + section) > 1000:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += "\n\n" + section if current_chunk else section
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    else:
        # For text and PDF, use paragraph-based chunking
        chunks = paragraph_chunk(text)
    
    # Filter out very small chunks
    chunks = [chunk for chunk in chunks if len(chunk.strip()) > 100]
    
    # If no good chunks found, do sentence-based chunking
    if not chunks:
        chunks = sentence_chunk(text)
    
    return chunks

def paragraph_chunk(text: str) -> List[str]:
    """Chunk text by paragraphs."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk + paragraph) > 800:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def sentence_chunk(text: str) -> List[str]:
    """Chunk text by sentences when paragraph chunking fails."""
    # Split by sentence endings
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) > 600:
            if current_chunk:
                chunks.append(current_chunk.strip() + ".")
            current_chunk = sentence
        else:
            current_chunk += ". " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip() + ".")
    
    return chunks

def save_chunks(chunks: List[str], filename: str) -> str:
    """Save chunks to a file in the chunks directory."""
    # Create chunks directory if it doesn't exist
    os.makedirs("chunks", exist_ok=True)
    
    chunk_file_path = os.path.join("chunks", f"{os.path.splitext(filename)[0]}_chunks.txt")
    
    with open(chunk_file_path, "w", encoding="utf-8") as f:
        f.write(f"File: {filename}\n")
        f.write(f"Total Chunks: {len(chunks)}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i+1}:\n{chunk}\n\n")
            f.write("-" * 30 + "\n\n")
    
    return chunk_file_path