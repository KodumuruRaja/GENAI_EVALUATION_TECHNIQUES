import os
from langchain_community.document_loaders import PyPDFLoader # Fixed: Uses standalone package to bypass community deprecation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from dotenv import load_dotenv

# DeepEval Imports
from deepeval.test_case import LLMTestCase
# FIXED: Changed AnswerRelevanceMetric to AnswerRelevancyMetric
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric 

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
docs = dense_retriever.invoke(query)  

print("\n--- Top Dense Search Results ---")
retrieved_contexts = []
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:\n{doc.page_content}")
    retrieved_contexts.append(doc.page_content)

# 5. Generate Answer using an LLM Chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
Answer:
""")

context_str = "\n\n".join(retrieved_contexts)
chain = prompt | llm | StrOutputParser()

print("\nGenerating pipeline answer...")
generated_answer = chain.invoke({"context": context_str, "question": query})
print(f"Generated Answer: {generated_answer}")

# =====================================================================
# 6. DEEPEVAL EVALUATION PIPELINE
# =====================================================================
print("\n--- Preparing DeepEval Metrics ---")

# DeepEval evaluates pipelines via a unified LLMTestCase object
test_case = LLMTestCase(
    input=query,
    actual_output=generated_answer,
    retrieval_context=retrieved_contexts
)

# Initialize metrics with explicit model anchoring
faithfulness_metric = FaithfulnessMetric(threshold=0.5, model="gpt-4o-mini")
# FIXED: Initialized the correct metric instance
answer_relevance_metric = AnswerRelevancyMetric(threshold=0.5, model="gpt-4o-mini")

# Measure individual metrics
faithfulness_metric.measure(test_case)
answer_relevance_metric.measure(test_case)

# Extract precise metric scores
faithfulness_score = faithfulness_metric.score
answer_relevance_score = answer_relevance_metric.score

# =====================================================================
# 7. FORMAT RESULT TO PANDAS DATAFRAME
# =====================================================================
evaluation_results = {
    "faithfulness": [faithfulness_score],
    "answer_relevance": [answer_relevance_score]
}

df_results = pd.DataFrame(evaluation_results)

print("\n--- DeepEval Evaluation Results ---")
print(df_results[["faithfulness", "answer_relevance"]])
