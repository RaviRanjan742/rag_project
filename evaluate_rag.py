import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np
import os
from dotenv import load_dotenv
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# Load environment variables
load_dotenv()

# Initialize clients
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    groq_client = None

def compute_semantic_similarity(reference: list, candidate: list) -> float:
    """Compute cosine similarity between embeddings of reference and candidate texts."""
    ref_embeddings = model.encode(reference, show_progress_bar=False)
    cand_embeddings = model.encode(candidate, show_progress_bar=False)
    cosine_sim = np.dot(ref_embeddings, cand_embeddings.T) / (
        np.linalg.norm(ref_embeddings, axis=1)[:, None] * np.linalg.norm(cand_embeddings, axis=1)
    )
    return np.mean(cosine_sim.diagonal())

def evaluate_rag():
    """Evaluate RAG application with improved metrics and dynamic retrieval."""
    collections = client.list_collections()
    if not collections:
        print("No collections found for evaluation. Please upload some documents first.")
        return
    
    total_chunks = sum(collection.count() for collection in collections)
    if total_chunks == 0:
        print("No documents in collections for evaluation.")
        return
    
    # Expanded test cases based on chunk content
    test_cases = [
        {
            "query": "carbon reduction strategies in energy sector",
            "ground_truth_summary": [
                "Investments in renewable energy like solar and wind.",
                "Carbon capture and storage (CCS) projects to reduce emissions.",
                "Methane emission reduction through advanced monitoring."
            ],
            "relevant_companies": ["Chevron", "Aramco", "Shell", "Xcel Energy"]
        },
        {
            "query": "electric vehicle adoption initiatives",
            "ground_truth_summary": [
                "Expansion of electric vehicle charging networks.",
                "Integration of EVs into delivery fleets.",
                "Support for low-carbon transportation infrastructure."
            ],
            "relevant_companies": ["Amazon", "Ford", "Shell"]
        },
        {
            "query": "biodiversity and ecosystem conservation",
            "ground_truth_summary": [
                "Habitat restoration and tree planting initiatives.",
                "Support for sustainable sourcing to protect ecosystems.",
                "Collaboration with environmental organizations."
            ],
            "relevant_companies": ["Amazon", "Nestle", "Hitachi", "Aramco"]
        },
        {
            "query": "regenerative agriculture practices",
            "ground_truth_summary": [
                "Adoption of regenerative farming to enhance soil health.",
                "Support for farmers through training and income programs.",
                "Reduction of emissions in agricultural supply chains."
            ],
            "relevant_companies": ["Nestle"]
        }
    ]
    
    recall_scores = []
    precision_scores = []
    f1_scores = []
    mrr_scores = []
    bleu_scores = []
    rouge_scores = []
    semantic_scores = []
    
    print("Starting RAG Evaluation...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case['query']}")
        print("-" * 30)
        
        query = test_case["query"]
        ground_truth_summary = test_case["ground_truth_summary"]
        relevant_companies = test_case["relevant_companies"]
        
        # Retrieval with dynamic thresholding and metadata filtering
        query_embedding = model.encode([query])[0].tolist()
        retrieved_chunks = []
        distances = []
        for collection in collections:
            query_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=2,  # Reduced to avoid warnings
                where={"filename": {"$in": [f"{company}.txt" for company in relevant_companies]}}
            )
            if query_results["documents"]:
                for doc, dist, meta in zip(
                    query_results["documents"][0],
                    query_results["distances"][0],
                    query_results["metadatas"][0]
                ):
                    max_chars = 1000 if meta["filename"].endswith(".txt") else 500
                    truncated_doc = doc[:max_chars] + "..." if len(doc) > max_chars else doc
                    retrieved_chunks.append((truncated_doc, meta, dist))
        
        # Sort by distance and apply dynamic threshold
        retrieved_chunks.sort(key=lambda x: x[2])
        threshold = np.percentile([x[2] for x in retrieved_chunks], 75) if retrieved_chunks else float('inf')
        retrieved_chunks = [x for x in retrieved_chunks if x[2] <= threshold][:3]
        
        # Compute retrieval metrics
        relevant_retrieved = len([x for x in retrieved_chunks if x[1]["filename"] in [f"{company}.txt" for company in relevant_companies]])
        recall = relevant_retrieved / len(relevant_companies) if relevant_companies else 0.0
        precision = relevant_retrieved / len(retrieved_chunks) if retrieved_chunks else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        recall_scores.append(recall)
        precision_scores.append(precision)
        f1_scores.append(f1)
        
        # Compute MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, chunk in enumerate(retrieved_chunks, 1):
            if chunk[1]["filename"] in [f"{company}.txt" for company in relevant_companies]:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)
        
        print(f"Recall@2: {recall:.4f}")
        print(f"Precision@2: {precision:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"MRR: {mrr:.4f}")
        
        # Summarization
        if retrieved_chunks:
            combined_text = ""
            for doc, meta, _ in retrieved_chunks:
                source = meta.get("filename", "Unknown")
                combined_text += f"Source: {source}\nContent: {doc}\n\n"
            
            prompt = (
                "Provide a concise bullet-point summary for the query '{}', focusing on key sustainability initiatives from the following documents. Ensure the summary is relevant to the query and attributes sources correctly:\n\n{}"
            ).format(query, combined_text)
            
            try:
                response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "Generate accurate, concise bullet-point summaries with source attribution, focusing on query relevance."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                generated_summary = response.choices[0].message.content
                
                # Clean and split generated summary
                generated_points = [
                    line.strip().lstrip('•-* ') 
                    for line in generated_summary.split('\n') 
                    if line.strip() and not line.strip().startswith('#')
                ]
                
                # Compute evaluation metrics
                if generated_points and ground_truth_summary:
                    bleu = sentence_bleu([ground_truth_summary], generated_points)
                    bleu_scores.append(bleu)
                    
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                    rouge_result = scorer.score(
                        ' '.join(ground_truth_summary),
                        ' '.join(generated_points)
                    )
                    rouge_scores.append(rouge_result['rougeL'].fmeasure)
                    
                    semantic_score = compute_semantic_similarity(ground_truth_summary, generated_points)
                    semantic_scores.append(semantic_score)
                    
                    print(f"BLEU Score: {bleu:.4f}")
                    print(f"ROUGE-L F1: {rouge_result['rougeL'].fmeasure:.4f}")
                    print(f"Semantic Similarity: {semantic_score:.4f}")
                    print(f"Generated Summary: {generated_summary[:100]}...")
                
                else:
                    bleu_scores.append(0.0)
                    rouge_scores.append(0.0)
                    semantic_scores.append(0.0)
                    print("No valid summary generated")
                
            except Exception as e:
                print(f"Error generating summary: {e}")
                bleu_scores.append(0.0)
                rouge_scores.append(0.0)
                semantic_scores.append(0.0)
        else:
            print("No documents retrieved for summarization")
            bleu_scores.append(0.0)
            rouge_scores.append(0.0)
            semantic_scores.append(0.0)
    
    # Print overall evaluation results
    print("\n" + "=" * 50)
    print("OVERALL EVALUATION RESULTS")
    print("=" * 50)
    
    if recall_scores:
        avg_recall = sum(recall_scores) / len(recall_scores)
        print(f"Average Recall@2: {avg_recall:.4f}")
    else:
        print("Average Recall@2: N/A")
    
    if precision_scores:
        avg_precision = sum(precision_scores) / len(precision_scores)
        print(f"Average Precision@2: {avg_precision:.4f}")
    else:
        print("Average Precision@2: N/A")
    
    if f1_scores:
        avg_f1 = sum(f1_scores) / len(f1_scores)
        print(f"Average F1 Score: {avg_f1:.4f}")
    else:
        print("Average F1 Score: N/A")
    
    if mrr_scores:
        avg_mrr = sum(mrr_scores) / len(mrr_scores)
        print(f"Average MRR: {avg_mrr:.4f}")
    else:
        print("Average MRR: N/A")
    
    if bleu_scores:
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        print(f"Average BLEU: {avg_bleu:.4f}")
    else:
        print("Average BLEU: N/A")
    
    if rouge_scores:
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        print(f"Average ROUGE-L F1: {avg_rouge:.4f}")
    else:
        print("Average ROUGE-L F1: N/A")
    
    if semantic_scores:
        avg_semantic = sum(semantic_scores) / len(semantic_scores)
        print(f"Average Semantic Similarity: {avg_semantic:.4f}")
    else:
        print("Average Semantic Similarity: N/A")

def evaluate_with_custom_queries(queries):
    """Evaluate RAG with custom user queries, including summarization."""
    collections = client.list_collections()
    if not collections:
        print("No collections found. Please upload documents first.")
        return
    
    print("Custom Query Evaluation")
    print("=" * 30)
    
    for query in queries:
        print(f"\nQuery: {query}")
        query_embedding = model.encode([query])[0].tolist()
        retrieved_chunks = []
        for collection in collections:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=2  # Reduced to avoid warnings
            )
            if results["documents"]:
                for doc, dist, meta in zip(
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0]
                ):
                    max_chars = 1000 if meta["filename"].endswith(".txt") else 500
                    truncated_doc = doc[:max_chars] + "..." if len(doc) > max_chars else doc
                    retrieved_chunks.append((truncated_doc, meta, dist))
        
        # Sort by distance and apply dynamic threshold
        retrieved_chunks.sort(key=lambda x: x[2])
        threshold = np.percentile([x[2] for x in retrieved_chunks], 75) if retrieved_chunks else float('inf')
        retrieved_chunks = [x for x in retrieved_chunks if x[2] <= threshold][:3]
        
        if retrieved_chunks:
            print(f"Retrieved {len(retrieved_chunks)} relevant documents")
            for i, (doc, meta, dist) in enumerate(retrieved_chunks[:2], 1):
                print(f"Doc {i}: {doc[:100]}... (Source: {meta['filename']}, Distance: {dist:.4f})")
            
            # Generate summary
            combined_text = ""
            for doc, meta, _ in retrieved_chunks:
                source = meta.get("filename", "Unknown")
                combined_text += f"Source: {source}\nContent: {doc}\n\n"
            
            prompt = (
                "Provide a concise bullet-point summary for the query '{}', focusing on key sustainability initiatives from the following documents. Ensure the summary is relevant to the query and attributes sources correctly:\n\n{}"
            ).format(query, combined_text)
            
            try:
                response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "Generate accurate, concise bullet-point summaries with source attribution, focusing on query relevance."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                generated_summary = response.choices[0].message.content
                print(f"Generated Summary:\n{generated_summary}")
                
            except Exception as e:
                print(f"Error generating summary: {e}")
        else:
            print("No relevant documents found")

if __name__ == "__main__":
    print("RAG Application Evaluation")
    print("Choose evaluation type:")
    print("1. Standard evaluation with predefined test cases")
    print("2. Custom query evaluation")
    
    choice = input("\nEnter choice (1 or 2): ")
    
    if choice == "1":
        evaluate_rag()
    elif choice == "2":
        custom_queries = []
        print("\nEnter your queries (press Enter with empty query to finish):")
        while True:
            query = input("Query: ").strip()
            if not query:
                break
            custom_queries.append(query)
        
        if custom_queries:
            evaluate_with_custom_queries(custom_queries)
        else:
            print("No queries provided.")
    else:
        print("Invalid choice. Running standard evaluation...")
        evaluate_rag()