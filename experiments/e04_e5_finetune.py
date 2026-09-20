"""Fine-tune multilingual E5 for direct query-to-item retrieval.

The validation query texts are excluded from supervised training. Evaluation
uses the same 50k-item catalog as E03 so dense and lexical scores are comparable.
Run from the repository root: python experiments/e04_e5_finetune.py
"""

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from e01_baseline import DATA, ITEM_COLS, QUERY_COLS, ROOT, validation_queries
from e03_e5_dense import MODEL_DIR, passage_text, recall_at


SEED = 42
MODEL_OUTPUT = ROOT / "models" / "e5-avito-e04"
ARTIFACTS = ROOT / "artifacts"


def query_text(rows: pd.DataFrame) -> list[str]:
    phrase = rows["search_query"].fillna("").astype(str)
    params = rows["search_infm_params_text"].fillna("").astype(str).str[:120]
    return ("query: " + phrase + ". " + params).tolist()


def validation_items(train: pd.DataFrame, truth: list[set[str]]) -> pd.DataFrame:
    items = train[ITEM_COLS].drop_duplicates("item_id").reset_index(drop=True)
    relevant = set().union(*truth)
    positives = items.loc[items["item_id"].isin(relevant)]
    negatives = items.loc[~items["item_id"].isin(relevant)].sample(
        n=50_000 - len(positives), random_state=SEED
    )
    return pd.concat([positives, negatives], ignore_index=True)


def training_pairs(train: pd.DataFrame, held_out: set[str], n_pairs: int) -> pd.DataFrame:
    pairs = train.loc[~train["search_query"].isin(held_out)]
    pairs = pairs.sample(frac=1, random_state=SEED)
    pairs = pairs.drop_duplicates("search_query").drop_duplicates("item_id")
    if len(pairs) < n_pairs:
        raise ValueError(f"Only {len(pairs)} unique query-item pairs remain")
    return pairs.sample(n=n_pairs, random_state=SEED + 1).reset_index(drop=True)


def train_model(model: SentenceTransformer, pairs: pd.DataFrame, batch_size: int) -> None:
    queries = query_text(pairs)
    passages = passage_text(pairs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    n_steps = (len(pairs) + batch_size - 1) // batch_size
    warmup = max(1, int(n_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, (n_steps - step) / max(1, n_steps - warmup))
    )
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pairs))
    model.train()
    started = time.monotonic()

    for step, start in enumerate(range(0, len(order), batch_size), 1):
        batch = order[start:start + batch_size]
        query_features = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in model.tokenize([queries[i] for i in batch]).items()}
        passage_features = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in model.tokenize([passages[i] for i in batch]).items()}
        q_emb = torch.nn.functional.normalize(model(query_features)["sentence_embedding"], dim=1)
        p_emb = torch.nn.functional.normalize(model(passage_features)["sentence_embedding"], dim=1)
        similarity = q_emb @ p_emb.T / 0.05
        target = torch.arange(len(batch), device=model.device)
        loss = (
            torch.nn.functional.cross_entropy(similarity, target)
            + torch.nn.functional.cross_entropy(similarity.T, target)
        ) / 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 50 == 0 or step == n_steps:
            print(f"Train step {step}/{n_steps}, loss={loss.item():.4f}, elapsed={time.monotonic() - started:.0f}s", flush=True)


def retrieve(model: SentenceTransformer, queries: pd.DataFrame, items: pd.DataFrame, vectors: np.ndarray, location_bonus: float = 0.0) -> list[list[str]]:
    query_vectors = model.encode(
        query_text(queries), batch_size=128, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    item_tensor = torch.as_tensor(vectors.astype(np.float32), device=model.device)
    item_ids = items["item_id"].to_numpy()
    item_locations = torch.as_tensor(items["item_location_id"].to_numpy().copy(), device=model.device)
    query_locations = torch.as_tensor(queries["search_location_id"].to_numpy().copy(), device=model.device)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(queries), 32):
            scores = torch.as_tensor(query_vectors[start:start + 32], device=model.device) @ item_tensor.T
            if location_bonus:
                scores += location_bonus * (query_locations[start:start + 32, None] == item_locations[None, :])
            indices = torch.topk(scores, k=200, dim=1).indices.cpu().numpy()
            predictions.extend(item_ids[indices].tolist())
    return predictions


def main(n_pairs: int, batch_size: int, evaluate_only: bool, full_catalog: bool) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    train = pd.read_parquet(DATA / "train.parquet", columns=QUERY_COLS + ITEM_COLS)
    queries, truth = validation_queries(train)
    items = (
        train[ITEM_COLS].drop_duplicates("item_id").reset_index(drop=True)
        if full_catalog else validation_items(train, truth)
    )
    vector_path = ARTIFACTS / f"e04_validation_items_{len(items)}.npy"
    if evaluate_only:
        model = SentenceTransformer(str(MODEL_OUTPUT), device=device, local_files_only=True)
    else:
        pairs = training_pairs(train, set(queries["search_query"]), n_pairs)
        print(f"Train pairs: {len(pairs)}; held-out queries: {len(queries)}; validation items: {len(items)}", flush=True)
        model = SentenceTransformer(str(MODEL_DIR), device=device, local_files_only=True)
        model.max_seq_length = 96
        train_model(model, pairs, batch_size)
        MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
        model.save(str(MODEL_OUTPUT))
        print(f"Saved model: {MODEL_OUTPUT}", flush=True)
    model.eval()
    if vector_path.exists():
        vectors = np.load(vector_path, mmap_mode="r")
    else:
        passages = passage_text(items)
        ARTIFACTS.mkdir(exist_ok=True)
        partial = vector_path.with_name(vector_path.stem + ".partial.npy")
        vectors = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(len(items), 768))
        for start in range(0, len(items), 2048):
            stop = min(start + 2048, len(items))
            vectors[start:stop] = model.encode(
                passages[start:stop], batch_size=128, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            ).astype(np.float16)
            if stop % 10_240 == 0 or stop == len(items):
                vectors.flush()
                print(f"Encoded validation items: {stop}/{len(items)}", flush=True)
        del vectors, passages
        os.replace(partial, vector_path)
        vectors = np.load(vector_path, mmap_mode="r")
    del train
    predictions = retrieve(model, queries, items, vectors)
    for k in (50, 100, 200):
        print(f"Fine-tuned E5 Recall@{k}: {recall_at(predictions, truth, k):.6f}", flush=True)
    if evaluate_only:
        scores = np.array([len(set(row[:50]) & relevant) / len(relevant) for row, relevant in zip(predictions, truth)])
        groups = {
            "empty filter": queries["search_infm_params_text"].eq("").to_numpy(),
            "nonempty filter": queries["search_infm_params_text"].ne("").to_numpy(),
            "query with digits": queries["search_query"].str.contains(r"\d", regex=True).to_numpy(),
        }
        for name, mask in groups.items():
            print(f"{name}: {scores[mask].mean():.6f} ({mask.sum()} queries)", flush=True)
        print(f"Zero-hit queries: {(scores == 0).sum()}/{len(scores)}", flush=True)
        lookup = items.set_index("item_id")["item_title_raw"].to_dict()
        for i in np.flatnonzero(scores == 0)[:10]:
            relevant = [lookup[item_id] for item_id in truth[i]]
            retrieved = [lookup[item_id] for item_id in predictions[i][:3]]
            print(f"MISS {queries.iloc[i]['search_query']!r} | gold={relevant[:3]} | top3={retrieved}", flush=True)
        for bonus in (0.02, 0.05, 0.10, 0.20):
            adjusted = retrieve(model, queries, items, vectors, location_bonus=bonus)
            print(f"Location bonus {bonus:.2f} Recall@50: {recall_at(adjusted, truth, 50):.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--full-catalog", action="store_true")
    args = parser.parse_args()
    if args.full_catalog and not args.evaluate:
        parser.error("--full-catalog requires --evaluate")
    main(args.pairs, args.batch_size, args.evaluate, args.full_catalog)
