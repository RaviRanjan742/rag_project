import streamlit as st
from file_processor import process_file, save_chunks
from embedding_manager import create_embeddings, query_vector_db, load_processed_files, save_processed_files
import os

# Streamlit UI
st.title("Sustainability RAG Application ")
st.write("Upload files (.txt, .pdf, .csv, .json) to process and query across all stored content.")

# Initialize session state and load processed files
if "processed_files" not in st.session_state:
    st.session_state.processed_files = load_processed_files()

# File uploader
uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf", "csv", "json"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Skip if file already processed
        if uploaded_file.name in st.session_state.processed_files:
            st.info(f"{uploaded_file.name} already processed. Skipping.")
            continue
        
        # Process file and get chunks
        with st.spinner(f"Processing {uploaded_file.name}..."):
            chunks = process_file(uploaded_file)
            if chunks:
                # Save chunks to file
                chunk_file_path = save_chunks(chunks, uploaded_file.name)
                st.success(f"Chunks saved to {chunk_file_path}")
                
                # Create and store embeddings
                create_embeddings(chunks, uploaded_file.name)
                st.session_state.processed_files.add(uploaded_file.name)
                save_processed_files(st.session_state.processed_files)
                st.success(f"Embeddings stored for {uploaded_file.name}")
            else:
                st.error(f"Failed to process {uploaded_file.name}")

# # Display processed files
# if st.session_state.processed_files:
#     st.subheader("Processed Files")
#     for file_name in st.session_state.processed_files:
#         st.write(f"- {file_name}")

# Clear all data button
if st.button("Clear All Processed Files"):
    if st.session_state.processed_files:
        from embedding_manager import clear_all_data
        clear_all_data()
        st.session_state.processed_files.clear()
        save_processed_files(st.session_state.processed_files)
        st.success("All processed files and embeddings cleared!")
        st.rerun()

# Query interface
st.subheader("Query All Stored Content")
query = st.text_input("Enter your query:")
if query:
    with st.spinner("Searching..."):
        results = query_vector_db(query)
        if results:
            st.markdown("### Summarized Results")
            st.markdown(results)
        else:
            st.markdown("No relevant results found or an error occurred.")

# Create necessary directories
if not os.path.exists("chunks"):
    os.makedirs("chunks")
if not os.path.exists("data"):
    os.makedirs("data")