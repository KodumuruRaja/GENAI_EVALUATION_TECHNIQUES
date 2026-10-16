# ✨ Generative AI Evaluation Techniques

<div align="center">
  <img src="https://img.shields.io/badge/AI-Evaluation-8A2BE2?style=for-the-badge&logo=OpenAI&logoColor=white" alt="AI Evaluation" />
  <img src="https://img.shields.io/badge/RAG-Ready-00C2A8?style=for-the-badge" alt="RAG Ready" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
</div>

<p align="center">
  <strong>Standalone Python examples for evaluating Retrieval-Augmented Generation (RAG) workflows</strong>
</p>

This repository contains a curated set of Python scripts for measuring how well a retrieval pipeline and answer-generation system performs. Each example focuses on a different evaluation metric and can be run independently.

## 🌟 What this repo includes

- 📚 Retrieval quality evaluation for RAG systems
- 🧠 LLM-as-a-judge patterns for semantic evaluation
- 📊 Classic metrics such as BLEU and ROUGE
- 🔍 Ranking metrics like Hit Rate and MRR
- 🛰️ Observability integrations with LangSmith, Langfuse, and TruLens
- 🧪 Standalone scripts designed for experimentation and learning

## 🧩 Example scripts

| Script | Metric / Focus | What it demonstrates |
| --- | --- | --- |
| `BLEU_Score.py` | BLEU-1 to BLEU-4 | Compares retrieved text with a reference answer |
| `Rouge_Score.py` | ROUGE-1, ROUGE-2, ROUGE-L | Measures overlap between retrieved text and a reference answer |
| `Context_Precision.py` | Context Precision | Uses an LLM judge to score retrieved chunks |
| `Context_Recall.py` | Context Recall | Checks whether retrieved context covers reference facts |
| `Hit_rate.py` | Hit Rate @ K | Evaluates whether relevant chunks appear in top results |
| `MRR.py` | Mean Reciprocal Rank | Measures the rank of the first relevant result |
| `Faithfulness_and_Answer_relevance.py` | Faithfulness + Answer Relevance | Custom LLM-as-a-judge evaluation patterns |
| `DeepEval.py` | DeepEval metrics | Faithfulness and answer relevance scoring |
| `Langsmith.py` | LangSmith evaluation | Dataset, tracing, and correctness scoring |
| `Langfuse.py` | Langfuse tracing | Experiment tracking and correctness evaluation |
| `TruLens.py` | TruLens metrics | Context relevance, groundedness, and answer relevance |

## 🛠️ Repository purpose

The examples are intentionally independent scripts rather than a single application. Each script contains a sample question and reference answer you can customize for your own dataset and use case.

> Update the question, reference answer, and PDF path in each script to evaluate your own retrieval pipeline.

## ✅ Requirements

Before running the scripts, make sure you have:

- Python 3.10 or newer
- An OpenAI API key
- A PDF file to evaluate
- Additional credentials for LangSmith and/or Langfuse examples

### Install dependencies

```bash
python -m pip install \
  langchain-community \
  langchain-text-splitters \
  langchain-chroma \
  langchain-openai \
  langchain-core \
  pypdf \
  chromadb \
  python-dotenv \
  pandas \
  nltk \
  rouge-score \
  deepeval \
  langsmith \
  langfuse \
  trulens
```

## ⚙️ Configuration

1. Place a PDF file somewhere accessible on your machine.
2. In the script you want to run, update the `pdf_loader(...)` path. The current files reference a machine-specific path that is not included in this repository.
3. Create a `.env` file in the project root with your OpenAI key:

```dotenv
OPENAI_API_KEY=your-openai-api-key
```

4. For `Langsmith.py`, also add your LangSmith key:

```dotenv
LANGSMITH_API_KEY=your-langsmith-api-key
```

5. For `Langfuse.py`, configure your Langfuse credentials:

```dotenv
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

> Never commit `.env` files or real API keys to version control.

## ▶️ Run an example

From the repository root, run any script individually:

```bash
python Context_Precision.py
```

Replace the filename with a script from the table above. Each script prints scores and intermediate results to the terminal. The LangSmith and Langfuse examples also report experiments or traces to their respective platforms.

## 📝 Notes

- The PDF, chunk size, chunk overlap, retrieval count, query, and reference answer are configured directly inside each script.
- Several metrics use an LLM as a judge, so results depend on the model, prompt, and evaluation data.
- Treat scores as useful evaluation signals rather than absolute truths.
- BLEU and ROUGE examples compare retrieved text with reference text and do not directly evaluate the generated answer.

## 🚀 Why use these evaluations?

When building RAG systems, it is important to measure more than just whether the model can answer a question. These examples help you assess:

- whether the retrieved context is relevant
- whether the answer is grounded in the source material
- whether the correct evidence appears early in retrieval results
- how different evaluation methods behave for your domain and dataset

If you are experimenting with RAG and LLM evaluation, this repository is a practical starting point for understanding commonly used metrics and tooling.

---

<p align="center">
  Built for <strong>Generative AI + Evaluation</strong> workflows
</p>
