"""Sparse lexical retrieval for the first Recall@50 submission.

Run from the repository root: python experiments/e01_baseline.py
The train interactions supply validation labels and a validation item corpus;
no interaction labels are used to fit the TF-IDF indexes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "NLP_avito_interns-dataset"
QUERY_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
ITEM_COLS = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_location_id",
]
SEED = 42
TOP_K = 50


def item_text(items: pd.DataFrame) -> list[str]:
    # Titles contain the service name. Descriptions and parameters add specific
    # terms, but are truncated to limit boilerplate and index size.
    title = items["item_title_raw"].fillna("").astype(str)
    params = items["item_infm_params_text"].fillna("").astype(str).str[:180]
    description = items["item_description_raw"].fillna("").astype(str).str[:500]
    return (title + " " + title + " " + params + " " + description).tolist()


def retrieve(queries: pd.DataFrame, items: pd.DataFrame) -> list[list[str]]:
    documents = item_text(items)
    query_word = (
        queries["search_query"].fillna("").astype(str)
        + " "
        + queries["search_infm_params_text"].fillna("").astype(str)
    ).tolist()
    query_char = queries["search_query"].fillna("").astype(str).tolist()

    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=400_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_items = word.fit_transform(documents)
    word_queries = word.transform(query_word)
    del documents

    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=350_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_items = char.fit_transform(items["item_title_raw"].fillna("").astype(str))
    char_queries = char.transform(query_char)
    del char

    item_ids = items["item_id"].to_numpy()
    item_locations = items["item_location_id"].to_numpy()
    query_locations = queries["search_location_id"].to_numpy()
    predictions = []

    for start in range(0, len(queries), 16):
        stop = min(start + 16, len(queries))
        scores = 0.7 * (word_queries[start:stop] @ word_items.T).toarray()
        scores += 0.3 * (char_queries[start:stop] @ char_items.T).toarray()
        # 83% of train clicks have matching location IDs. Keep the global
        # candidates: some relevant items have a different ID or wider reach.
        scores *= 1 + 0.4 * (
            query_locations[start:stop, None] == item_locations[None, :]
        )
        top = np.argpartition(scores, -TOP_K, axis=1)[:, -TOP_K:]
        top = np.take_along_axis(
            top, np.argsort(np.take_along_axis(scores, top, axis=1), axis=1)[:, ::-1], axis=1
        )
        predictions.extend(item_ids[top].tolist())
        if stop % 256 == 0 or stop == len(queries):
            print(f"Retrieved {stop}/{len(queries)} queries", flush=True)

    return predictions


def validation_queries(train: pd.DataFrame) -> tuple[pd.DataFrame, list[set[str]]]:
    # Benchmark query texts are unique, and its filter is empty in 63% of rows.
    # Draw distinct texts, matching that filter mix, so repeated train queries
    # cannot make local validation artificially easy.
    contexts = train[QUERY_COLS].drop_duplicates()
    contexts = contexts.loc[
        contexts["search_category"].eq(114)
        & contexts["search_is_delivery_search"].eq(0)
    ]
    empty = contexts.loc[contexts["search_infm_params_text"].eq("")]
    empty = empty.drop_duplicates("search_query").sample(n=630, random_state=SEED)
    filled = contexts.loc[
        contexts["search_infm_params_text"].ne("")
        & ~contexts["search_query"].isin(empty["search_query"])
    ]
    filled = filled.drop_duplicates("search_query").sample(n=370, random_state=SEED + 1)
    queries = pd.concat([empty, filled], ignore_index=True)

    # Group by selected context while preserving the sampled query order.
    positives = train[QUERY_COLS + ["item_id"]].drop_duplicates().merge(
        queries.reset_index(names="validation_id"), on=QUERY_COLS, how="inner", sort=False
    )
    truth = positives.groupby("validation_id")["item_id"].agg(lambda x: set(x)).reindex(
        range(len(queries))
    )
    return queries, truth.tolist()


def main() -> None:
    train = pd.read_parquet(DATA / "train.parquet", columns=QUERY_COLS + ITEM_COLS)
    queries, truth = validation_queries(train)
    validation_items = train[ITEM_COLS].drop_duplicates("item_id").reset_index(drop=True)
    print(f"Validation: {len(queries)} queries, {len(validation_items)} items", flush=True)
    validation_predictions = retrieve(queries, validation_items)
    recall = np.mean(
        [len(set(prediction) & relevant) / len(relevant) for prediction, relevant in zip(validation_predictions, truth)]
    )
    print(f"Validation Recall@50: {recall:.6f}", flush=True)
    for label, mask in [
        ("empty filter", queries["search_infm_params_text"].eq("")),
        ("nonempty filter", queries["search_infm_params_text"].ne("")),
    ]:
        values = [
            len(set(validation_predictions[j]) & truth[j]) / len(truth[j])
            for j in np.flatnonzero(mask.to_numpy())
        ]
        print(f"{label}: {np.mean(values):.6f} ({len(values)} queries)", flush=True)

    del train, validation_items

    benchmark_queries = pd.read_parquet(DATA / "benchmark_queries.parquet")
    benchmark_items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=ITEM_COLS)
    print(
        f"Benchmark: {len(benchmark_queries)} queries, {len(benchmark_items)} items",
        flush=True,
    )
    predictions = retrieve(benchmark_queries, benchmark_items)
    answer = pd.DataFrame(
        {
            "query_id": benchmark_queries["query_id"],
            "answer": [" ".join(row) for row in predictions],
        }
    )
    assert answer["query_id"].is_unique
    assert answer["query_id"].str.len().eq(16).all()
    valid_ids = set(benchmark_items["item_id"])
    assert all(len(row) == TOP_K and len(set(row)) == TOP_K for row in predictions)
    assert all(set(row) <= valid_ids for row in predictions)
    output = ROOT / "submissions" / "submission_01.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(output, index=False)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
