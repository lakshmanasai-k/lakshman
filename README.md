# Lakshmana Sai Kalakanda

### AI/ML Engineer | Generative AI | RAG

I build Python-based AI and machine learning applications, with a focus on retrieval-augmented generation (RAG), document assistants, and data-driven systems. I’m interested in making AI answers useful, traceable to source documents, and constrained by clear guardrails.

## Technical Skills

- **Languages:** Python, SQL
- **Generative AI:** RAG, prompt engineering, embeddings, vector search, retrieval evaluation, citations
- **Frameworks and APIs:** LangChain, LangGraph, FastAPI, REST APIs
- **Machine learning:** scikit-learn, XGBoost, Random Forest, Logistic Regression, feature engineering, model evaluation
- **Data:** Pandas, NumPy, PySpark, ETL
- **Search and cloud:** Azure AI Search, AWS S3, SageMaker, RDS, CloudWatch
- **LLM APIs:** OpenAI, Gemini, Claude

## Certifications

- AWS Certified Machine Learning Engineer – Associate (MLA-C01)
- AWS Certified Solutions Architect – Associate (SAA-C03)

## Selected Projects

### End-to-End RAG Application

A document question-answering system that ingests policy files, prepares searchable chunks, creates embeddings, retrieves relevant passages, and generates answers grounded in those sources.

- Process PDF, DOCX, TXT, and Markdown files; split content into overlapping chunks.
- Create embeddings and rank passages with semantic similarity.
- Orchestrate retrieval, evidence routing, and answer generation with LangGraph; serve requests through FastAPI.
- Include source citations and a fallback when the selected documents do not support an answer.

### Internal HR Policy Assistant

A RAG use case for answering employee questions about topics such as leave, benefits, workplace guidelines, and remote-work policies. Responses should cite the applicable policy passages and identify the policy version when available.

### Insurance Policy Assistant

A RAG use case for searching insurance documents and answering questions about policy terms, coverage, exclusions, deductibles, eligibility, and claims steps. Guardrails keep answers tied to retrieved policy text, flag missing or conflicting information, and avoid making claim or coverage decisions.

### Credit Card Fraud Detection

A machine learning workflow for identifying potentially suspicious card transactions.

- Prepare transaction data and preserve a stratified train, validation, and test split.
- Compare Logistic Regression and Random Forest, with validation-based threshold selection.
- Evaluate precision, recall, F1 score, ROC-AUC, and a confusion matrix, with attention to missed fraud and false alerts.

## RAG Guardrails and Evaluation

- Ground answers in retrieved documents and include citations.
- Treat retrieved document text as reference data, separate from system instructions.
- Validate response structure and qualify answers when evidence is missing.
- Evaluate retrieval relevance, answer grounding, and citation accuracy.

## Project Code and Data Engineering

- [Project index and Data Engineer responsibilities](projects/PROJECTS.md)
- [Complete HR and Insurance Policy RAG application](projects/complete_policy_rag_application.py)
- [Credit Card Fraud Detection code](projects/credit_card_fraud_detection.py)
- [Finance Databricks Medallion pipeline](projects/finance_databricks_pipeline.py)
- [Finance data handling and Databricks Data Engineer guide](projects/finance_databricks_data_handling.md)
- [Python dependencies for the examples](projects/requirements.txt)

This repository is my portfolio space for AI/ML, RAG, and data engineering work.
