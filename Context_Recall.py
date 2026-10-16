import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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
# Define the expected correct answer from your benchmark data
ground_truth_answer = "The document covers the India’s long struggle for independence from British colonial rule, highlighting reform movements, cultural influences, key leaders, and major events that shaped the freedom movement." 

docs = dense_retriever.invoke(query)  

print("\n--- Top Dense Search Results ---")
all_retrieved_text = ""
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:\n{doc.page_content}")
    all_retrieved_text += doc.page_content + "\n"

# 5. Calculate Context Recall via Custom LLM Evaluator
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step A: Deconstruct ground truth answer into distinct factual statements
fact_extraction_prompt = f"""
Analyze the given Ground Truth Answer and break it down into an explicit list of individual, atomic factual statements.
Respond with a JSON object containing a single key "statements" which holds a list of strings.

Ground Truth Answer: {ground_truth_answer}

JSON Response:
"""

print("\n--- Extracting Truth Statements ---")
extraction_response = llm.invoke(fact_extraction_prompt)
try:
    clean_extraction = extraction_response.content.strip().replace("```json", "").replace("```", "")
    statements = json.loads(clean_extraction).get("statements", [])
except Exception as e:
    print(f"Failed to parse statements: {e}")
    statements = [ground_truth_answer]

print(f"Extracted Facts to Verify: {statements}")

# Step B: Check if each fact exists semantically in the retrieved vector context
verified_facts_count = 0

print("\n--- Verifying Facts Against Dense Chunks ---")
for statement in statements:
    verification_prompt = f"""
    You are an evaluation judge. Determine if the given 'Truth Statement' can be semantically inferred or found directly within the 'Retrieved Context'.
    Respond with a JSON object containing a single key "verified" with a value of either true or false.

    Truth Statement: {statement}
    Retrieved Context: {all_retrieved_text}

    JSON Response:
    """
    
    verification_response = llm.invoke(verification_prompt)
    try:
        clean_verification = verification_response.content.strip().replace("```json", "").replace("```", "")
        is_verified = json.loads(clean_verification).get("verified", False)
    except Exception:
        is_verified = False
        
    print(f"Fact: '{statement}' -> Found in Dense Context: {is_verified}")
    if is_verified:
        verified_facts_count += 1

# Calculate Context Recall
if len(statements) == 0:
    context_recall_score = 0.0
else:
    context_recall_score = verified_facts_count / len(statements)

print("\n--- Evaluation Metrics ---")
print(f"Calculated Dense Context Recall Score: {context_recall_score:.4f}")
