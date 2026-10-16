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

# 4. Query the Dense Retriever
query = "What is the main topic of the document?"
docs = dense_retriever.invoke(query)  

print("\n--- Top Dense Search Results ---")
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:\n{doc.page_content}")

# 5. Calculate Context Precision via Custom LLM Judge
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("\n--- Evaluating Context Precision ---")
relevance_scores = []
running_relevant_count = 0
precision_at_k = []

for i, doc in enumerate(docs):
    precision_prompt = f"""
    You are an evaluation judge. Given a user query and a retrieved text chunk, determine if the chunk contains relevant information to answer the query.
    Respond with a JSON object containing a single key "relevant" with a value of either true or false.

    Query: {query}
    Retrieved Chunk: {doc.page_content}

    JSON Response:
    """
    try:
        response = llm.invoke(precision_prompt)
        clean_content = response.content.strip().replace("```json", "").replace("```", "")
        is_relevant = json.loads(clean_content).get("relevant", False)
    except Exception:
        is_relevant = False
        
    print(f"Chunk {i+1} Relevant: {is_relevant}")
    
    # Calculate Precision at position K
    if is_relevant:
        running_relevant_count += 1
        relevance_scores.append(1)
        precision_at_k.append(running_relevant_count / (i + 1))
    else:
        relevance_scores.append(0)
        precision_at_k.append(0)

# Apply Context Precision Formula
total_relevant = sum(relevance_scores)
context_precision_score = 0.0 if total_relevant == 0 else sum(p * r for p, r in zip(precision_at_k, relevance_scores)) / total_relevant

# =====================================================================
# FINAL METRIC OUTPUT
# =====================================================================
print("\n=== FINAL RETRIEVAL EVALUATION ===")
print(f"Context Precision Score : {context_precision_score:.4f}")
print("==================================")
