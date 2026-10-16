import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rouge_score import rouge_scorer
from dotenv import load_dotenv

load_dotenv()

def pdf_loader(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    return data

def chunk_documents(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# 1. Load and chunk documents
documents = pdf_loader(r"C:\Users\Megha\Desktop\GENAI_CONCEPTS\Chunking_Strategies\India's.pdf")
print(f"Successfully loaded {len(documents)} pages.")

chunks1 = chunk_documents(documents, chunk_size=100, chunk_overlap=20)
print(f"Number of chunks created: {len(chunks1)}")

# 2. Initialize Embeddings and Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
persist_directory = "./chroma_db"

vector_store = Chroma.from_documents(
    documents=chunks1,
    embedding=embeddings,
    persist_directory=persist_directory
)
print("Chunks successfully added to ChromaDB!")

# 3. Expose the vector store as a Dense Retriever
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 4. Query the Dense Retriever and build context pool
query = "What is the main topic of the document?"
# Define the expected correct benchmark answer
ground_truth_answer = "The document covers the economic growth, digital infrastructure, and development strategies of India." 

docs = dense_retriever.invoke(query)  

print("\n--- Top Dense Search Results ---")
all_retrieved_text = ""
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:\n{doc.page_content}")
    all_retrieved_text += doc.page_content + " "

# Clean up whitespace for evaluation
all_retrieved_text = all_retrieved_text.strip()

# 5. Calculate ROUGE Scores
# Initialize the scorer for ROUGE-1 (unigrams), ROUGE-2 (bigrams), and ROUGE-L (longest common subsequence)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(prediction=all_retrieved_text, target=ground_truth_answer)

# =====================================================================
# FINAL ROUGE METRICS OUTPUT
# =====================================================================
print("\n=== FINAL ROUGE EVALUATION ===")
print(f"ROUGE-1 (Precision: {scores['rouge1'].precision:.4f}, Recall: {scores['rouge1'].recall:.4f}, F1-Score: {scores['rouge1'].fmeasure:.4f})")
print(f"ROUGE-2 (Precision: {scores['rouge2'].precision:.4f}, Recall: {scores['rouge2'].recall:.4f}, F1-Score: {scores['rouge2'].fmeasure:.4f})")
print(f"ROUGE-L (Precision: {scores['rougeL'].precision:.4f}, Recall: {scores['rougeL'].recall:.4f}, F1-Score: {scores['rougeL'].fmeasure:.4f})")
print("==============================")
