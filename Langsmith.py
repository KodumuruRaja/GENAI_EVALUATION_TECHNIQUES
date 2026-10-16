import os
from langchain_community.document_loaders import PyPDFLoader # FIXED: Swapped to standalone package to bypass community deprecation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# LangSmith core client and evaluation runners
from langsmith import Client
from langsmith.evaluation import evaluate

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
# 5. LANGSMITH DATASET SETUP
# =====================================================================
print("\n--- Initializing LangSmith Client & Dataset ---")
client = Client()
dataset_name = "India PDF Evaluation Dataset"

if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Evaluation dataset for testing RAG chunking strategies.",
    )
    client.create_examples(
        inputs=[{"question": "What is the main topic of the document?"}],
        outputs=[{"ground_truth": "The document covers the economic growth, digital infrastructure, and development strategies of India."}],
        dataset_id=dataset.id,
    )
    print(f"Dataset '{dataset_name}' created successfully with test examples!")
else:
    print(f"Dataset '{dataset_name}' already exists. Using existing examples.")

# =====================================================================
# 6. DEFINING THE LLM-AS-A-JUDGE EVALUATOR FUNCTIONS
# =====================================================================
# FIXED: Replaced non-existent LangChainStringEvaluator with direct Python evaluation functions
def evaluate_correctness(run, example):
    """Evaluates if the model output matches the correct ground truth semantics."""
    student_answer = run.outputs.get("output")
    reference_answer = example.outputs.get("ground_truth")
    
    eval_prompt = f"""
    You are an evaluation judge. Compare the Student Answer against the Reference Answer for correctness.
    Determine if the Student Answer contains the same core facts as the Reference Answer.
    Respond with a score between 0.0 (completely wrong) and 1.0 (perfectly correct).
    Output only a raw float score number, nothing else.

    Reference Answer: {reference_answer}
    Student Answer: {student_answer}
    Score:"""
    
    score_response = llm.invoke(eval_prompt)
    try:
        score = float(score_response.content.strip())
    except ValueError:
        score = 0.0
    return {"key": "correctness", "score": score}

# Wrapper to pass inputs smoothly into the target RAG execution chain
def predict_rag(example):
    response = rag_chain.invoke(example["question"])
    return {"output": response}

# =====================================================================
# 7. EXECUTE EXPERIMENT RUN
# =====================================================================
print("\n--- Running LangSmith Experiment and Evaluation ---")
experiment_results = evaluate(
    predict_rag,
    data=dataset_name,
    evaluators=[evaluate_correctness],
    experiment_prefix="chroma-dense-retrieval",
)

print("\n=== LANGSMITH EVALUATION TASK COMPLETE ===")
print("You can view live evaluation metrics, execution times, and complete token trace trees at:")
print(f"https://langchain.com")
