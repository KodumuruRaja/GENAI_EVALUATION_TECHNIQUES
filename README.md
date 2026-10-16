# Generative AI Evaluation Techniques

This repository contains standalone Python examples for evaluating retrieval-augmented generation (RAG) workflows. The examples load a PDF, split it into chunks, index the chunks in a local Chroma vector store, and then calculate an evaluation metric or send traces and evaluations to an observability platform.

## Examples

| Script | What it demonstrates |
| --- | --- |
| `BLEU_Score.py` | BLEU-1 through BLEU-4, comparing retrieved text with a reference answer |
| `Rouge_Score.py` | ROUGE-1, ROUGE-2, and ROUGE-L, comparing retrieved text with a reference answer |
| `Context_Precision.py` | Context precision using an LLM judge to score retrieved chunks |
| `Context_Recall.py` | Context recall using an LLM judge to check reference facts against retrieved context |
| `Hit_rate.py` | Hit rate at K using an LLM judge |
| `MRR.py` | Mean reciprocal rank using an LLM judge |
| `Faithfulness_and_Answer_relevance.py` | Custom LLM-as-a-judge examples for faithfulness and answer relevance |
| `DeepEval.py` | DeepEval faithfulness and answer relevancy metrics |
| `Langsmith.py` | LangSmith dataset, experiment, and correctness evaluator |
| `Langfuse.py` | Langfuse tracing and correctness scoring |
| `TruLens.py` | TruLens context relevance, groundedness, and answer relevance metrics |

The examples are independent scripts, not a single application. They use a sample question and reference answer defined in the scripts; update these values to evaluate your own data.

## Requirements

- Python 3.10 or newer
- An OpenAI API key
- A PDF file to evaluate
- Additional service credentials for the LangSmith and Langfuse examples

Install the packages used across the examples:

```bash
python -m pip install langchain-community langchain-text-splitters langchain-chroma langchain-openai langchain-core pypdf chromadb python-dotenv pandas nltk rouge-score deepeval langsmith langfuse trulens-core trulens-apps-langchain trulens-providers-openai
```

## Configuration

1. Place a PDF file somewhere accessible on your computer.
2. In the script you want to run, update the PDF path in the `pdf_loader(...)` call. The current scripts refer to a machine-specific path that is not included in this repository.
3. Create a `.env` file in the repository root with your OpenAI key:

   ```dotenv
   OPENAI_API_KEY=your-openai-api-key
   ```

4. For `Langsmith.py`, configure a LangSmith API key as well:

   ```dotenv
   LANGSMITH_API_KEY=your-langsmith-api-key
   ```

5. For `Langfuse.py`, configure your Langfuse project keys. `LANGFUSE_HOST` is optional and defaults to the Langfuse cloud URL:

   ```dotenv
   LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
   LANGFUSE_SECRET_KEY=your-langfuse-secret-key
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

Do not commit `.env` files or real API keys.

## Run an example

From the repository root, run one script at a time:

```bash
python Context_Precision.py
```

Replace the filename with any example from the table above. The scripts print scores and intermediate results to the terminal. The LangSmith and Langfuse examples also send experiment or trace data to their configured services. The scripts create or use a local `chroma_db` directory for the vector store.

`BLEU_Score.py` may download the NLTK `punkt_tab` tokenizer data the first time it runs.

## Notes

- The PDF, chunk size, chunk overlap, retrieval count, query, and reference answer are configured directly in each script.
- Several metrics use an LLM as a judge, so results depend on the model, prompt, and data. Treat scores as evaluation signals, not guarantees of correctness.
- The BLEU and ROUGE examples compare retrieved text with reference text; they do not evaluate the generated answer.
