import os
import warnings
# Swapped to standalone package to eliminate community deprecation alerts
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
from dotenv import load_dotenv

# TruLens Core Infrastructure Components
from trulens.core import TruSession
from trulens.core import Metric # Updated from legacy Feedback to clear DeprecationWarnings
from trulens.apps.langchain import TruChain 
from trulens.providers.openai import OpenAI as TruOpenAI 

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
# 5. MODERN TRULENS V1.0+ OTEL METRICS SETUP
# =====================================================================
print("\n--- Initializing TruLens Session & Metric Engines ---")
session = TruSession()
session.reset_database() 

# Initialize TruLens LLM provider context engine
tru_openai = TruOpenAI(model_engine="gpt-4o-mini")

# FIXED: Replaced old .on() lens selectors with modern OTEL string path dictionaries
# 1. Context Relevance
m_context_relevance = (
    Metric(
        name="Context Relevance",
        implementation=tru_openai.context_relevance_with_cot_reasons
    )
    .on_input()
    .on_output("$.context") # Targets the extracted context variable mapping inside your chain dict
)

# 2. Groundedness / Faithfulness
m_groundedness = (
    Metric(
        name="Groundedness",
        implementation=tru_openai.groundedness_measure_with_cot_reasons
    )
    .on_input("$.context") # Evaluates grounded text bounds relative to your extracted context strings
    .on_output()
)

# 3. Answer Relevance
m_answer_relevance = (
    Metric(
        name="Answer Relevance",
        implementation=tru_openai.relevance_with_cot_reasons
    )
    .on_input() # Maps straight back to user query inputs
    .on_output() # Compares against final string answer generation boundaries
)

# Wrap the chain inside the TruChain instrument recorder wrapper
tru_recorder = TruChain(
    rag_chain,
    app_id="Chroma_Dense_RAG",
    metrics=[m_context_relevance, m_groundedness, m_answer_relevance] # Changed from feedbacks to metrics
)

# =====================================================================
# 6. RUN RETRIEVAL PIPELINE WITH TRULENS INSTRUMENTATION
# =====================================================================
query = "What is the main topic of the document?"
print(f"\nInvoking chain under TruLens recorder for query: '{query}'")

with tru_recorder as recording:
    generated_answer = rag_chain.invoke(query)

print(f"Generated Answer: {generated_answer}")

# =====================================================================
# 7. EXTRACT TRULENS METRICS AS A PANDAS DATAFRAME
# =====================================================================
print("\nWaiting for TruLens asynchronous evaluation to finish...")
session.force_flush() 

records, feedback_defs = session.get_records_and_feedback(app_ids=["Chroma_Dense_RAG"])

print("\n--- TruLens Evaluation Results ---")
# Filter and display metrics cleanly
metric_columns = [col for col in records.columns if col.startswith("metric_") or col == "app_id"]
if metric_columns:
    print(records[metric_columns])
else:
    # Fallback to display everything if columns names are generic
    print(records)
