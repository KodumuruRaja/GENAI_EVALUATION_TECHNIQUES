import os
import warnings
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# Swapped to standalone package to resolve langchain-community deprecation warnings
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Automatically download tokenizer if missing
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

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
ground_truth_answer = "The document covers the economic growth, digital infrastructure, and development strategies of India." 

docs = dense_retriever.invoke(query)  

print("\n--- Top Dense Search Results ---")
all_retrieved_text = ""
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:\n{doc.page_content}")
    all_retrieved_text += doc.page_content + " "

all_retrieved_text = all_retrieved_text.strip()

# 5. Calculate BLEU Score
reference_tokens = [nltk.word_tokenize(ground_truth_answer.lower())]
candidate_tokens = nltk.word_tokenize(all_retrieved_text.lower())

smoothing = SmoothingFunction().method1

# FIXED: Removed the invalid 'candidate' keyword argument
bleu_score = sentence_bleu(
    references=reference_tokens,
    hypothesis=candidate_tokens,
    smoothing_function=smoothing
)

bleu_1 = sentence_bleu(reference_tokens, candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
bleu_2 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
bleu_3 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
bleu_4 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

print("\n=== FINAL BLEU EVALUATION ===")
print(f"Cumulative BLEU Score (BLEU-4): {bleu_score:.4f}")
print(f"Individual BLEU-1 (Unigram)   : {bleu_1:.4f}")
print(f"Individual BLEU-2 (Bigram)    : {bleu_2:.4f}")
print(f"Individual BLEU-3 (Trigram)   : {bleu_3:.4f}")
print(f"Individual BLEU-4 (4-gram)    : {bleu_4:.4f}")
print("==============================")
