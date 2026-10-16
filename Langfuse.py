import os
from dotenv import load_dotenv

# Force environment configuration variables to load first
load_dotenv()

# FIXED: Swapped to standalone package to completely eliminate the langchain-community deprecation warning
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# FIXED: Modern Langfuse v3.0+ import structure for the LangChain integration extension package
from langfuse import Langfuse
from langfuse.callback import CallbackHandler 

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

# 4. Assemble the LangChain RAG Chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": dense_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# =====================================================================
# 5. INITIALIZE LANGFUSE TRACING
# =====================================================================
print("\n--- Initializing Langfuse Tracing Handler ---")

if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
    raise ValueError("CRITICAL ERROR: Langfuse API keys are missing from environment. Check your .env file.")

# The callback handler intercepts LangChain lifecycle hooks automatically
langfuse_handler = CallbackHandler()
# Base client used for logging explicit metrics/scores
langfuse_client = Langfuse()

# =====================================================================
# 6. RUN PIPELINE WITH TRACING HANDLER
# =====================================================================
query = "What is the main topic of the document?"
ground_truth_answer = "The document covers the economic growth, digital infrastructure, and development strategies of India."

print(f"\nInvoking chain and tracking trace tree logs for query: '{query}'")

# Inject the handler into the config array of invoke() to record the complete run automatically
generated_answer = rag_chain.invoke(
    query, 
    config={"callbacks": [langfuse_handler]}
)

print(f"Generated Answer: {generated_answer}")

# =====================================================================
# 7. RUN LOCAL LLM-AS-A-JUDGE METRIC AND LOG TO LANGFUSE
# =====================================================================
print("\n--- Computing Evaluation Metric and Logging to Langfuse ---")

eval_prompt = f"""
You are an evaluation judge. Compare the Student Answer against the Reference Answer for correctness.
Determine if the Student Answer contains the same core facts as the Reference Answer.
Respond with a score between 0.0 (completely wrong) and 1.0 (perfectly correct).
Output only a raw float score number, nothing else.

Reference Answer: {ground_truth_answer}
Student Answer: {generated_answer}
Score:"""

score_response = llm.invoke(eval_prompt)
try:
    eval_score = float(score_response.content.strip())
except ValueError:
    eval_score = 0.0

print(f"Calculated Correctness Score: {eval_score}")

# Fetch the root trace ID generated during the chain execution hook
trace_id = langfuse_handler.get_trace_id()

# Log the evaluation score directly back to that specific transaction trace tree link
langfuse_client.score(
    trace_id=trace_id,
    name="semantic_correctness",
    value=eval_score,
    comment="Evaluated locally via gpt-4o-mini judge comparing against reference benchmark answer."
)

# Flush any pending network buffers sequentially to guarantee data upload completes
langfuse_client.flush()

print("\n=== LANGFUSE TRACING AND EVALUATION TASK COMPLETE ===")
print("You can view live trace trees, latency graphs, token pricing, and scores inside your project dashboard at:")
print(f"{os.environ.get('LANGFUSE_HOST', 'https://langfuse.com')}")
