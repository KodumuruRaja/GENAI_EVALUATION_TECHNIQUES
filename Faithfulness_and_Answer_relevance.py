import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
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
# 6. MANUAL RAGAS METRICS (FAITHFULNESS & ANSWER RELEVANCE)
# =====================================================================
print("\n--- Running Custom LLM-as-a-Judge Evaluation ---")

# --- A. CALCULATE FAITHFULNESS (Is the answer grounded in context?) ---
# Step 1: Extract individual claims from the generated answer
claims_prompt = f"""
Analyze the given Generated Answer and break it down into an explicit list of individual, atomic factual claims.
Respond with a JSON object containing a single key "claims" which holds a list of strings.

Generated Answer: {generated_answer}

JSON Response:
"""
try:
    claims_response = llm.invoke(claims_prompt)
    clean_claims = claims_response.content.strip().replace("```json", "").replace("```", "")
    claims = json.loads(clean_claims).get("claims", [])
except Exception:
    claims = [generated_answer]

# Step 2: Check if each claim is supported by the retrieved context
verified_claims_count = 0
for claim in claims:
    verify_prompt = f"""
    Determine if the given 'Factual Claim' can be completely inferred from or is directly supported by the 'Retrieved Context'.
    Respond with a JSON object containing a single key "supported" with a value of either true or false.

    Factual Claim: {claim}
    Retrieved Context: {context_str}

    JSON Response:
    """
    try:
        verify_resp = llm.invoke(verify_prompt)
        clean_verify = verify_resp.content.strip().replace("```json", "").replace("```", "")
        is_supported = json.loads(clean_verify).get("supported", False)
    except Exception:
        is_supported = False
    
    if is_supported:
        verified_claims_count += 1

faithfulness_score = 0.0 if len(claims) == 0 else verified_claims_count / len(claims)


# --- B. CALCULATE ANSWER RELEVANCE (Does the answer match the question?) ---
# Step 1: Ask the LLM to generate a hypothetical question that matches the generated answer
reverse_question_prompt = f"""
Given the following Generated Answer, write the most natural, direct question that this text is answering. 
Respond with a JSON object containing a single key "question" which holds the string.

Generated Answer: {generated_answer}

JSON Response:
"""
try:
    reverse_resp = llm.invoke(reverse_question_prompt)
    clean_reverse = reverse_resp.content.strip().replace("```json", "").replace("```", "")
    generated_question = json.loads(clean_reverse).get("question", "")
except Exception:
    generated_question = query

# Step 2: Calculate similarity between original query and generated question using semantic embedding overlap
orig_embedding = embeddings.embed_query(query)
gen_embedding = embeddings.embed_query(generated_question)

# Compute Cosine Similarity manually
dot_product = sum(a * b for a, b in zip(orig_embedding, gen_embedding))
magnitude_orig = sum(a * a for a in orig_embedding) ** 0.5
magnitude_gen = sum(b * b for b in gen_embedding) ** 0.5
answer_relevance_score = dot_product / (magnitude_orig * magnitude_gen)


# =====================================================================
# 7. FORMAT RESULT TO PANDAS DATAFRAME
# =====================================================================
evaluation_results = {
    "faithfulness": [faithfulness_score],
    "answer_relevance": [answer_relevance_score]
}

df_results = pd.DataFrame(evaluation_results)

print("\n--- Ragas Evaluation Results ---")
print(df_results[["faithfulness", "answer_relevance"]])
