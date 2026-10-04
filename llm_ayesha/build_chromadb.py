import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# PATHS


PAIR_FILE = "fass_pair_chunks.csv"
FALLBACK_FILE = "fass_fallback_chunks.csv"
DB_PATH = "./drug_db"
COLLECTION_NAME = "medication_interactions"


# LOAD DATA


print("Loading chunk files...")

pair_df = pd.read_csv(PAIR_FILE)
fallback_df = pd.read_csv(FALLBACK_FILE)

print("Pair chunks:", len(pair_df))
print("Fallback chunks:", len(fallback_df))



# PREPARE PAIR DOCUMENTS


pair_ready = pd.DataFrame({
    "document": pair_df["chunk_text"].fillna(""),
    "chunk_type": "interaction_pair",

    "drug_1_name": pair_df["drug_1_name"].fillna(""),
    "drug_1_atc": pair_df["drug_1_atc"].fillna(""),

    "drug_2_name": pair_df["drug_2_name"].fillna(""),
    "drug_2_atc": pair_df["drug_2_atc"].fillna(""),

    "severity": pair_df["severity"].fillna("INFO"),

    "source_url": pair_df["source_url"].fillna("")
})



# PREPARE FALLBACK DOCUMENTS


fallback_ready = pd.DataFrame({
    "document": fallback_df["chunk_text"].fillna(""),
    "chunk_type": "interaction_statement",

    "drug_1_name": fallback_df["drug_name"].fillna(""),
    "drug_1_atc": fallback_df["atc_code"].fillna(""),

    "drug_2_name": "",
    "drug_2_atc": "",

    "severity": "UNCLASSIFIED",

    "source_url": fallback_df["source_url"].fillna("")
})



# COMBINE


all_docs = pd.concat(
    [pair_ready, fallback_ready],
    ignore_index=True
)

print("Total documents:", len(all_docs))



# LOAD EMBEDDING MODEL


print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Embedding model loaded.")



# CREATE CHROMADB


client = chromadb.PersistentClient(
    path=DB_PATH
)

# Delete old collection if it exists
try:
    client.delete_collection(COLLECTION_NAME)
    print("Old collection deleted.")
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME
)


# CREATE EMBEDDINGS


documents = all_docs["document"].tolist()

print("Creating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True,
    normalize_embeddings=True,
    batch_size=32
)

print("Embeddings created:", embeddings.shape)



# PREPARE METADATA


metadatas = all_docs[
    [
        "chunk_type",
        "drug_1_name",
        "drug_1_atc",
        "drug_2_name",
        "drug_2_atc",
        "severity",
        "source_url"
    ]
].to_dict(orient="records")

ids = [
    f"chunk_{i}"
    for i in range(len(all_docs))
]



# ADD TO CHROMADB IN BATCHES


BATCH_SIZE = 500

print("Adding documents to ChromaDB...")

for start in range(0, len(documents), BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        len(documents)
    )

    collection.add(
        ids=ids[start:end],
        documents=documents[start:end],
        embeddings=embeddings[start:end].tolist(),
        metadatas=metadatas[start:end]
    )

    print(
        f"Added {end}/{len(documents)}"
    )



# FINAL CHECK


print()
print("ChromaDB build complete.")
print("Collection:", COLLECTION_NAME)
print("Documents stored:", collection.count())
print("Database path:", DB_PATH)
