# Project Code Documents

This folder contains learning and portfolio examples for AI/ML and data engineering. The policy RAG app supports separate HR and insurance document domains. The finance pipeline demonstrates Databricks medallion data handling.

## Projects

- [Complete HR and Insurance Policy RAG Application](./complete_policy_rag_application.py) — PDF/DOCX/TXT ingestion, chunking, embeddings, domain-filtered retrieval, source references, and answer guardrails.
- [Credit Card Fraud Detection](./credit_card_fraud_detection.py) — class-imbalance-aware model comparison, validation-based threshold selection, and precision/recall-focused evaluation.
- [Finance Databricks Medallion Pipeline](./finance_databricks_pipeline.py) — Auto Loader ingestion to Bronze, validation and quarantine in Silver, and aggregated reporting in Gold.

## Data Engineer Responsibilities Shown

A data engineer on these examples would:

1. Ingest files and records from approved sources and preserve source lineage.
2. Define schemas, handle schema changes, and quarantine malformed data.
3. Apply quality checks for required fields, valid types, uniqueness, and reconciliation.
4. Organize data into Bronze, Silver, and Gold Delta tables.
5. Use Unity Catalog permissions and governed volumes to control access.
6. Monitor jobs, checkpoints, failures, data freshness, and processing costs.
7. Document ownership, runbooks, dependencies, and recovery steps.

## Finance Data Handling

The finance example expects a tokenized card reference from an approved upstream system. It does not process raw card numbers, CVV, or PIN values. Restrict row-level Bronze, Silver, and quarantine tables; use least-privilege Unity Catalog grants; keep sensitive fields out of logs; and publish only the aggregates needed by reporting users.

For real HR, insurance, or finance data, use approved documents and datasets only. The RAG example is a learning demo: add authentication, per-user authorization, persistent storage, audit controls, and retention rules before production use.
