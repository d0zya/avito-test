"""Full-catalog semantic retrieval with multilingual E5 (no fine-tuning).

Model files must be available locally in models/multilingual-e5-base. The
script first evaluates the same 1000 queries as E01; --benchmark encodes the
benchmark catalog and writes a submission only after the local result is known.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from e01_baseline import (
    DATA, ITEM_COLS, QUERY_COLS, ROOT, TOP_K,
    retrieve as lexical_retrieve, validation_queries,
)


MODEL_DIR = ROOT / "models" / "multilingual-e5-base"
ARTIFACTS = ROOT / "artifacts"
MAX_LENGTH = 96
TOP_CANDIDATES = 200


def passage_text(items: pd.DataFrame) -> list[str]:
    title = items["item_title_raw"].fillna("").astype(str)
    params = items["item_infm_params_text"].fillna("").astype(str).str[:120]
    description = items["item_description_raw"].fillna("").astype(str).str[:180]
    return ("passage: " + title + ". " + params + ". " + description).tolist()


def encode_items(model: SentenceTransformer, items: pd.DataFrame, name: str) -> np.ndarray:
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / f"e03_{name}_e5base_96.npy"
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        if cached.shape == (len(items), 768):
            print(f"Using cached embeddings: {path}", flush=True)
            return cached

    texts = passage_text(items)
    partial = path.with_name(path.stem + ".partial.npy")
    vectors = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float16, shape=(len(items), 768)
    )
    for start in range(0, len(texts), 2048):
        stop = min(start + 2048, len(texts))
        embedding = model.encode(
            texts[start:stop],
            batch_size=128,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors[start:stop] = embedding.astype(np.float16)
        if stop % 20_480 == 0 or stop == len(texts):
            vectors.flush()
            print(f"Encoded {stop}/{len(texts)} items", flush=True)
    del vectors, texts
    os.replace(partial, path)
    return np.load(path, mmap_mode="r")


def retrieve(
    model: SentenceTransformer,
    queries: pd.DataFrame,
    items: pd.DataFrame,
    item_vectors: np.ndarray,
) -> list[list[str]]:
    query_vectors = model.encode(
        ("query: " + queries["search_query"].fillna("").astype(str)).tolist(),
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    device = model.device
    item_tensor = torch.as_tensor(np.asarray(item_vectors, dtype=np.float32), device=device)
    item_ids = items["item_id"].to_numpy()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(queries), 32):
            stop = min(start + 32, len(queries))
            batch = torch.as_tensor(query_vectors[start:stop], device=device)
            scores = batch @ item_tensor.T
            indices = torch.topk(scores, k=TOP_CANDIDATES, dim=1).indices.cpu().numpy()
            predictions.extend(item_ids[indices].tolist())
            if stop % 256 == 0 or stop == len(queries):
                print(f"Retrieved {stop}/{len(queries)} queries", flush=True)
    return predictions


def recall_at(predictions: list[list[str]], truth: list[set[str]], k: int) -> float:
    return float(
        np.mean(
            [len(set(row[:k]) & relevant) / len(relevant) for row, relevant in zip(predictions, truth)]
        )
    )


def main(make_benchmark: bool, sample_items: int | None) -> None:
    if make_benchmark and sample_items:
        raise ValueError("--sample-items is only for validation")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    model = SentenceTransformer(str(MODEL_DIR), device=device, local_files_only=True)
    model.max_seq_length = MAX_LENGTH
    assert model.get_sentence_embedding_dimension() == 768

    train = pd.read_parquet(DATA / "train.parquet", columns=QUERY_COLS + ITEM_COLS)
    queries, truth = validation_queries(train)
    items = train[ITEM_COLS].drop_duplicates("item_id").reset_index(drop=True)
    if sample_items:
        relevant = set().union(*truth)
        positive_items = items.loc[items["item_id"].isin(relevant)]
        negatives = items.loc[~items["item_id"].isin(relevant)].sample(
            n=sample_items - len(positive_items), random_state=42
        )
        items = pd.concat([positive_items, negatives], ignore_index=True)
    del train
    vectors = encode_items(model, items, f"validation_items_{len(items)}")
    predictions = retrieve(model, queries, items, vectors)
    for k in (50, 100, 200):
        print(f"E5 Recall@{k}: {recall_at(predictions, truth, k):.6f}", flush=True)
    if sample_items:
        lexical = lexical_retrieve(queries, items)
        print(f"E01 Recall@50: {recall_at(lexical, truth, 50):.6f}", flush=True)
        union = [list(set(dense[:50]) | set(sparse[:50])) for dense, sparse in zip(predictions, lexical)]
        print(f"Oracle union of E5/E01 top50: {recall_at(union, truth, 100):.6f}", flush=True)
        for dense_quota in (10, 20, 30, 40):
            hybrid = []
            for dense, sparse in zip(predictions, lexical):
                ranking = dict.fromkeys(dense[:dense_quota] + sparse + dense)
                hybrid.append(list(ranking)[:50])
            print(f"Hybrid {dense_quota} E5 slots: {recall_at(hybrid, truth, 50):.6f}", flush=True)
    if not make_benchmark:
        return

    benchmark_queries = pd.read_parquet(DATA / "benchmark_queries.parquet")
    benchmark_items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=ITEM_COLS)
    vectors = encode_items(model, benchmark_items, "benchmark_items")
    predictions = retrieve(model, benchmark_queries, benchmark_items, vectors)
    output = ROOT / "submissions" / "submission_03.csv"
    valid_ids = set(benchmark_items["item_id"])
    assert all(len(set(row[:TOP_K])) == TOP_K for row in predictions)
    assert all(set(row[:TOP_K]) <= valid_ids for row in predictions)
    pd.DataFrame(
        {
            "query_id": benchmark_queries["query_id"],
            "answer": [" ".join(row[:TOP_K]) for row in predictions],
        }
    ).to_csv(output, index=False)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sample-items", type=int)
    args = parser.parse_args()
    main(make_benchmark=args.benchmark, sample_items=args.sample_items)
