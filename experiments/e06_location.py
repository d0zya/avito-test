"""Single-stage location-aware retrieval for the time-limited second submission."""

import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from e01_baseline import DATA, ITEM_COLS, QUERY_COLS, ROOT, TOP_K, item_text, validation_queries


LOCATION_WEIGHTS = (0.4, 1.0, 2.0, 4.0)


def retrieve(queries: pd.DataFrame, items: pd.DataFrame, weights: tuple[float, ...]) -> dict[float, list[list[str]]]:
    word = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=400_000,
        sublinear_tf=True, dtype=np.float32,
    )
    word_items = word.fit_transform(item_text(items))
    word_queries = word.transform(
        (queries["search_query"].fillna("").astype(str) + " " +
         queries["search_infm_params_text"].fillna("").astype(str)).tolist()
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3,
        max_features=350_000, sublinear_tf=True, dtype=np.float32,
    )
    char_items = char.fit_transform(items["item_title_raw"].fillna("").astype(str))
    char_queries = char.transform(queries["search_query"].fillna("").astype(str))
    item_ids = items["item_id"].to_numpy()
    item_locations = items["item_location_id"].to_numpy()
    query_locations = queries["search_location_id"].to_numpy()
    predictions = {weight: [] for weight in weights}

    for start in range(0, len(queries), 16):
        stop = min(start + 16, len(queries))
        scores = 0.7 * (word_queries[start:stop] @ word_items.T).toarray()
        scores += 0.3 * (char_queries[start:stop] @ char_items.T).toarray()
        same_location = query_locations[start:stop, None] == item_locations[None, :]
        for weight in weights:
            adjusted = scores * (1 + weight * same_location)
            top = np.argpartition(adjusted, -TOP_K, axis=1)[:, -TOP_K:]
            top = np.take_along_axis(
                top, np.argsort(np.take_along_axis(adjusted, top, axis=1), axis=1)[:, ::-1], axis=1
            )
            predictions[weight].extend(item_ids[top].tolist())
        if stop % 256 == 0 or stop == len(queries):
            print(f"Retrieved {stop}/{len(queries)} queries", flush=True)
    return predictions


def main(make_submission: bool) -> None:
    train = pd.read_parquet(DATA / "train.parquet", columns=QUERY_COLS + ITEM_COLS)
    queries, truth = validation_queries(train)
    items = train[ITEM_COLS].drop_duplicates("item_id").reset_index(drop=True)
    del train
    predictions = retrieve(queries, items, LOCATION_WEIGHTS)
    results = {}
    for weight, rows in predictions.items():
        per_query = np.array([len(set(row) & relevant) / len(relevant) for row, relevant in zip(rows, truth)])
        results[weight] = per_query.mean()
        print(f"location_weight={weight:.1f}: Recall@50={per_query.mean():.6f}", flush=True)
    selected = max(results, key=results.get)
    print(f"Selected weight: {selected:.1f}", flush=True)
    if not make_submission or results[selected] <= results[0.4]:
        return
    benchmark_queries = pd.read_parquet(DATA / "benchmark_queries.parquet")
    benchmark_items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=ITEM_COLS)
    rows = retrieve(benchmark_queries, benchmark_items, (selected,))[selected]
    ids = set(benchmark_items["item_id"])
    assert all(len(row) == TOP_K and len(set(row)) == TOP_K and set(row) <= ids for row in rows)
    answer = pd.DataFrame({
        "query_id": benchmark_queries["query_id"].astype(str),
        "answer": [" ".join(row) for row in rows],
    })
    assert answer["query_id"].is_unique
    assert answer["query_id"].str.len().eq(16).all()
    output = ROOT / "submissions" / "submission_02_location.csv"
    answer.to_csv(output, index=False)
    answer.to_csv(ROOT / "answer.csv", index=False)
    print(f"Saved {output} and answer.csv", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-submission", action="store_true")
    main(parser.parse_args().make_submission)
