# Lakshmana Sai Kalakanda

### AI/ML Engineer | Generative AI | RAG

I build Python-based AI and machine learning applications, with a focus on retrieval-augmented generation (RAG), document assistants, and data-driven systems. I’m interested in making AI answers useful, traceable to source documents, and constrained by clear guardrails.

## Technical Skills

- **Languages:** Python, SQL
- **Generative AI:** RAG, prompt engineering, embeddings, vector search, hybrid retrieval, citations
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

- Process PDF and DOCX files; clean and split content into meaningful chunks.
- Store embeddings and document metadata for retrieval.
- Combine semantic/vector search with keyword search to find relevant passages.
- Orchestrate retrieval and answer generation with LangGraph and serve requests through FastAPI.
- Include source citations, structured response validation, and a fallback when the documents do not support an answer.

### Internal HR Policy Assistant

A RAG use case for answering employee questions about topics such as leave, benefits, workplace guidelines, and remote-work policies. Responses should cite the applicable policy passages and identify the policy version when available.

### Insurance Policy Assistant

A RAG use case for searching insurance documents and answering questions about policy terms, coverage, exclusions, deductibles, eligibility, and claims steps. Guardrails keep answers tied to retrieved policy text, flag missing or conflicting information, and avoid making claim or coverage decisions.

### Credit Card Fraud Detection

A machine learning workflow for identifying potentially suspicious card transactions.

- Clean transaction data and build useful transaction features.
- Address class imbalance and compare models such as Logistic Regression, Random Forest, and XGBoost.
- Evaluate with precision, recall, F1 score, ROC-AUC, and a confusion matrix, with attention to missed fraud and false alerts.
- Structure the workflow so scoring can be exposed through an API and monitored over time.

## RAG Guardrails and Evaluation

- Ground answers in retrieved documents and include citations.
- Detect prompt-injection attempts and keep system instructions separate from document content.
- Validate response structure and decline or qualify answers when evidence is missing.
- Evaluate retrieval relevance, answer grounding, and citation accuracy.

This repository is my portfolio space for AI/ML, RAG, and data engineering work.
