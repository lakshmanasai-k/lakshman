"""
Finance transaction pipeline using Databricks Auto Loader, Delta Lake, and PySpark.

Run in a Databricks workspace after creating the required Unity Catalog schemas
and landing/operations volumes. Input files must already contain a tokenized
card reference; this pipeline must never ingest raw PAN, CVV, or PIN values.

The silver/gold writes are full refreshes for this learning example. Use
incremental MERGE/upsert logic and tested recovery procedures for production.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


spark = SparkSession.builder.getOrCreate()

CATALOG = "main"
BRONZE_TABLE = f"{CATALOG}.bronze.finance_transactions"
SILVER_TABLE = f"{CATALOG}.silver.finance_transactions"
QUARANTINE_TABLE = f"{CATALOG}.quarantine.invalid_finance_transactions"
GOLD_TABLE = f"{CATALOG}.gold.daily_finance_summary"

# Use governed Unity Catalog volumes for source, schema, and checkpoint paths.
LANDING_PATH = "/Volumes/main/bronze/finance_ops/landing"
SCHEMA_PATH = "/Volumes/main/bronze/finance_ops/schema"
CHECKPOINT_PATH = "/Volumes/main/bronze/finance_ops/checkpoints/transactions"


def ensure_schemas() -> None:
    for schema in ("bronze", "silver", "gold", "quarantine"):
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")


def ingest_bronze() -> None:
    """
    Incrementally ingest new CSV files to a raw Delta table.
    Bronze preserves source fields and adds ingestion lineage metadata.
    """
    stream = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", SCHEMA_PATH)
        .option("cloudFiles.schemaEvolutionMode", "rescue")
        .option("rescuedDataColumn", "_rescued_data")
        .option("header", "true")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(LANDING_PATH)
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
    )

    (
        stream.writeStream.format("delta")
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(availableNow=True)
        .toTable(BRONZE_TABLE)
        .awaitTermination()
    )


def build_silver() -> None:
    """
    Normalize names and types, quarantine invalid rows, and retain only the
    tokenized card reference required for restricted downstream use.
    """
    bronze = spark.table(BRONZE_TABLE)

    normalized = bronze.select(
        F.trim(F.col("transaction_id").cast("string")).alias("transaction_id"),
        F.to_timestamp(F.col("transaction_ts")).alias("transaction_ts"),
        F.col("amount").cast("decimal(18,2)").alias("amount"),
        F.upper(F.trim(F.col("currency").cast("string"))).alias("currency"),
        F.trim(F.col("merchant_id").cast("string")).alias("merchant_id"),
        F.trim(F.col("tokenized_card_ref").cast("string")).alias("tokenized_card_ref"),
        F.col("_source_file"),
        F.col("_ingested_at"),
        F.col("_rescued_data"),
    )

    invalid_condition = (
        F.col("transaction_id").isNull()
        | F.col("transaction_ts").isNull()
        | F.col("amount").isNull()
        | (F.col("amount") < F.lit(0))
        | F.col("currency").isNull()
        | (F.length(F.col("currency")) != 3)
        | F.col("merchant_id").isNull()
        | F.col("tokenized_card_ref").isNull()
        | F.col("_rescued_data").isNotNull()
    )

    reason = (
        F.when(F.col("transaction_id").isNull(), "missing_transaction_id")
        .when(F.col("transaction_ts").isNull(), "invalid_transaction_ts")
        .when(F.col("amount").isNull(), "invalid_amount")
        .when(F.col("amount") < F.lit(0), "negative_amount")
        .when(F.col("currency").isNull(), "missing_currency")
        .when(F.length(F.col("currency")) != 3, "invalid_currency")
        .when(F.col("merchant_id").isNull(), "missing_merchant_id")
        .when(F.col("tokenized_card_ref").isNull(), "missing_tokenized_card_ref")
        .otherwise("rescued_schema_data")
    )

    invalid_rows = normalized.filter(invalid_condition).withColumn(
        "_dq_reason", reason
    )
    valid_rows = (
        normalized.filter(~invalid_condition)
        .withColumn("transaction_date", F.to_date("transaction_ts"))
        .drop("_rescued_data")
    )

    # Quarantine may contain sensitive metadata; restrict access in Unity Catalog.
    (
        invalid_rows.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(QUARANTINE_TABLE)
    )
    (
        valid_rows.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(SILVER_TABLE)
    )


def build_gold() -> None:
    """
    Publish aggregate metrics for reporting without exposing card-level rows.
    """
    silver = spark.table(SILVER_TABLE)
    gold = (
        silver.groupBy("transaction_date", "currency", "merchant_id")
        .agg(
            F.count("*").alias("transaction_count"),
            F.round(F.sum("amount"), 2).alias("total_amount"),
            F.countDistinct("tokenized_card_ref").alias("unique_card_count"),
        )
        .withColumn("_refreshed_at", F.current_timestamp())
    )

    (
        gold.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(GOLD_TABLE)
    )


def run_pipeline() -> None:
    ensure_schemas()
    ingest_bronze()
    build_silver()
    build_gold()


if __name__ == "__main__":
    run_pipeline()
