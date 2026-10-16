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

# 3. Expose the vector store as a Dense Retriever (Top K = 2)
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 4. Query the Dense Retriever
query = "What is the main topic of the document?"
docs = dense_retriever.invoke(query)  

print("\n--- Top Dense Search Results ---")
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:\n{doc.page_content}")

# 5. Calculate Hit Rate via Custom LLM Judge
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("\n--- Evaluating Hit Rate ---")
has_hit = False

for i, doc in enumerate(docs):
    hit_prompt = f"""
    You are an evaluation judge. Given a user query and a retrieved text chunk, determine if the chunk contains relevant information to answer the query.
    Respond with a JSON object containing a single key "relevant" with a value of either true or false.

    Query: {query}
    Retrieved Chunk: {doc.page_content}

    JSON Response:
    """
    try:
        response = llm.invoke(hit_prompt)
        clean_content = response.content.strip().replace("```json", "").replace("```", "")
        is_relevant = json.loads(clean_content).get("relevant", False)
    except Exception:
        is_relevant = False
        
    print(f"Chunk {i+1} Relevant: {is_relevant}")
    
    # Hit Rate condition: If at least one document is relevant, it's a hit!
    if is_relevant:
        has_hit = True
        # We can break early because finding one relevant document satisfies the hit criteria
        print("-> Found a relevant document. Hit achieved!")
        break

# Hit rate score for a single query scenario is either 1.0 (Hit) or 0.0 (Miss)
hit_rate_score = 1.0 if has_hit else 0.0

# =====================================================================
# FINAL METRIC OUTPUT
# =====================================================================
print("\n=== FINAL RETRIEVAL EVALUATION ===")
print(f"Hit Rate @K Score : {hit_rate_score:.4f}")
print("==================================")
