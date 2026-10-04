import chromadb
from sentence_transformers import SentenceTransformer


DB_PATH = "/teamspace/studios/this_studio/LLMSCAI_G_19/llm_ayesha/drug_db"
COLLECTION_NAME = "medication_interactions"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)



# LOAD MODEL


model = SentenceTransformer(MODEL_NAME)



# LOAD CHROMADB


client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)



# EXACT STRUCTURED PAIR SEARCH


def search_exact_pair(atc_a, atc_b):

    atc_a = str(atc_a).upper().strip()
    atc_b = str(atc_b).upper().strip()

    results = collection.get(
        where={
            "$or": [
                {
                    "$and": [
                        {"drug_1_atc": {"$eq": atc_a}},
                        {"drug_2_atc": {"$eq": atc_b}},
                        {"chunk_type": {"$eq": "interaction_pair"}}
                    ]
                },
                {
                    "$and": [
                        {"drug_1_atc": {"$eq": atc_b}},
                        {"drug_2_atc": {"$eq": atc_a}},
                        {"chunk_type": {"$eq": "interaction_pair"}}
                    ]
                }
            ]
        },
        include=[
            "documents",
            "metadatas"
        ]
    )

    formatted = []

    for i in range(len(results["documents"])):

        formatted.append({
            "rank": i + 1,
            "document": results["documents"][i],
            "metadata": results["metadatas"][i],
            "retrieval_source": "exact_structured_pair"
        })

    return formatted


# FALLBACK RERANKING


def rerank_fallback(results, drug_b):

    drug_b = str(drug_b).lower().strip()

    for item in results:

        text = item["document"].lower()

        # Exact Drug B mention gets priority
        if drug_b and drug_b in text:
            item["drug_b_bonus"] = 1
        else:
            item["drug_b_bonus"] = 0

    # First prioritize Drug B mention,
    # then semantic distance (smaller = better)
    results.sort(
        key=lambda x: (
            -x["drug_b_bonus"],
            x["distance"]
        )
    )

    for i, item in enumerate(results, start=1):
        item["rank"] = i

    return results



# FILTERED FALLBACK SEARCH

def search_fallback_filtered(
    drug_a,
    atc_a,
    drug_b,
    top_k=10,
    candidate_k=30
):

    query = f"{drug_a} {drug_b} interaction"

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=candidate_k,
        where={
            "$and": [
                {
                    "chunk_type": {
                        "$eq": "interaction_statement"
                    }
                },
                {
                    "drug_1_atc": {
                        "$eq": str(atc_a).upper().strip()
                    }
                }
            ]
        }
    )

    formatted = []

    for i in range(len(results["documents"][0])):

        formatted.append({
            "document":
                results["documents"][0][i],

            "metadata":
                results["metadatas"][0][i],

            "distance":
                results["distances"][0][i],

            "retrieval_source":
                "fallback_statement"
        })

    formatted = rerank_fallback(
        formatted,
        drug_b
    )

    return formatted[:top_k]



# FINAL HYBRID SEARCH


def hybrid_search(
    drug_a,
    atc_a,
    drug_b,
    atc_b,
    fallback_top_k=5
):

    # 1. Try exact structured pair first
    exact_results = search_exact_pair(
        atc_a,
        atc_b
    )

    if exact_results:
        return exact_results

    # 2. Otherwise use filtered + reranked fallback
    return search_fallback_filtered(
        drug_a=drug_a,
        atc_a=atc_a,
        drug_b=drug_b,
        top_k=fallback_top_k,
        candidate_k=30
    )



# SIMPLE TEST


if __name__ == "__main__":

    results = search_fallback_filtered(
        drug_a="Digoxin",
        atc_a="C01AA05",
        drug_b="metoklopramid",
        top_k=10,
        candidate_k=30
    )

    for result in results:

        print("=" * 100)

        print(
            "RANK:",
            result["rank"]
        )

        print(
            "SOURCE:",
            result["retrieval_source"]
        )

        print(
            "DISTANCE:",
            result["distance"]
        )

        print()

        print(
            result["document"]
        )
