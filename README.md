# SEMRAG: Semantic Graph-Based Retrieval & Question Answering on PDFs

SEMRAG is a Python-based system that performs semantic retrieval augmented generation (RAG) from PDF documents. It combines semantic embeddings, knowledge graph construction, and community-aware retrieval to provide accurate, context-grounded answers to user queries.

This implementation uses Dr. B.R. Ambedkar’s writings as the example corpus but can be adapted to any PDF-based dataset.

# SEMRAG System WorkFlow
          PDF
           ↓
          Sentence Split
           ↓
          Semantic Chunking (Algorithm 1)
           ↓
          Buffer Merge (token-aware)
           ↓
          Chunk Embeddings
           ↓
          NER + Relation Extraction
           ↓
          Knowledge Graph (Entities + Relations)
           ↓
          Community Detection (Louvain)
           ↓
          Local Graph RAG (Eq. 4)
          Global Graph RAG (Eq. 5)
           ↓
          LLM Prompt (Chunks + Entities + Communities)
           ↓
          Answer

# Hallucination Control
Ensures answers are grounded in retrieved context; does not hallucinate information not present in the PDF.

# Example queries
**Grounded (in-document) queries:**
- What was Ambedkar's view on caste discrimination?
- What did Ambedkar say about social justice and equality?
- What did Ambedkar say about labor and workers' rights?

**Hallucination-test (out-of-document) queries:**
- What are Ambedkar's views on quantum mechanics?
- What were Ambedkar's views on Mahatma Gandhi's assassination?
- What did Ambedkar say about time travel?

Queries unrelated to the PDF (e.g., "What are Ambedkar's views on quantum mechanics?") will return "No relevant information found in the document." ensuring no hallucinations.
Installation.

## Installation

```bash
git clone https://github.com/<your-username>/SEMRAG.git
cd SEMRAG

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt

python -m spacy download en_core_web_sm
Ensure Ollama is installed and running for local LLM inference.

## Repository Structure

SEMRAG/
│
├── Ambedkar_book.pdf         # Example PDF
├── demo.py                   # Demo script
├── semrag.py                 # Core functions and classes
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
