"""Test whether separating the query phrase from search filters improves Recall@50.

Run from the repository root: python experiments/e02_filter_fields.py
E01 is reproduced on the same holdout for a paired comparison. After reviewing
the result, add --make-submission to write the benchmark file.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from e01_baseline import DATA, ITEM_COLS, QUERY_COLS, ROOT, TOP_K, item_text, validation_queries


FILTER_WEIGHTS = (0.0, 0.1, 0.2)


def top_ids(scores: np.ndarray, item_ids: np.ndarray) -> list[list[str]]:
    top = np.argpartition(scores, -TOP_K, axis=1)[:, -TOP_K:]
    top = np.take_along_axis(
        top, np.argsort(np.take_along_axis(scores, top, axis=1), axis=1)[:, ::-1], axis=1
    )
    return item_ids[top].tolist()


def retrieve(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    filter_weights: tuple[float, ...],
    compare_e01: bool = False,
) -> dict[float | str, list[list[str]]]:
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=400_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_items = word.fit_transform(item_text(items))
    phrase = queries["search_query"].fillna("").astype(str).tolist()
    word_phrase = word.transform(phrase)
    if compare_e01:
        combined = (
            queries["search_query"].fillna("").astype(str)
            + " "
            + queries["search_infm_params_text"].fillna("").astype(str)
        ).tolist()
        word_e01 = word.transform(combined)

    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=350_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_items = char.fit_transform(items["item_title_raw"].fillna("").astype(str))
    char_queries = char.transform(phrase)

    filters = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=150_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    filter_items = filters.fit_transform(
        items["item_infm_params_text"].fillna("").astype(str).str[:450]
    )
    filter_queries = filters.transform(
        queries["search_infm_params_text"].fillna("").astype(str)
    )

    item_ids = items["item_id"].to_numpy()
    item_locations = items["item_location_id"].to_numpy()
    query_locations = queries["search_location_id"].to_numpy()
    predictions: dict[float | str, list[list[str]]] = {
        weight: [] for weight in filter_weights
    }
    if compare_e01:
        predictions["e01"] = []

    for start in range(0, len(queries), 16):
        stop = min(start + 16, len(queries))
        char_scores = (char_queries[start:stop] @ char_items.T).toarray()
        phrase_scores = 0.7 * (word_phrase[start:stop] @ word_items.T).toarray()
        phrase_scores += 0.3 * char_scores
        filter_scores = (filter_queries[start:stop] @ filter_items.T).toarray()
        location_factor = 1 + 0.4 * (
            query_locations[start:stop, None] == item_locations[None, :]
        )

        for weight in filter_weights:
            scores = ((1 - weight) * phrase_scores + weight * filter_scores) * location_factor
            predictions[weight].extend(top_ids(scores, item_ids))

        if compare_e01:
            e01_scores = 0.7 * (word_e01[start:stop] @ word_items.T).toarray()
            e01_scores += 0.3 * char_scores
            e01_scores *= location_factor
            predictions["e01"].extend(top_ids(e01_scores, item_ids))

        if stop % 256 == 0 or stop == len(queries):
            print(f"Retrieved {stop}/{len(queries)} queries", flush=True)

    return predictions


def recall_per_query(predictions: list[list[str]], truth: list[set[str]]) -> np.ndarray:
    return np.array(
        [len(set(row) & relevant) / len(relevant) for row, relevant in zip(predictions, truth)]
    )


def main(make_submission: bool = False) -> None:
    train = pd.read_parquet(DATA / "train.parquet", columns=QUERY_COLS + ITEM_COLS)
    queries, truth = validation_queries(train)
    items = train[ITEM_COLS].drop_duplicates("item_id").reset_index(drop=True)
    print(f"Validation: {len(queries)} queries, {len(items)} items", flush=True)
    predictions = retrieve(queries, items, FILTER_WEIGHTS, compare_e01=True)
    del train, items

    baseline = recall_per_query(predictions["e01"], truth)
    assert np.isclose(baseline.mean(), 0.4088547619047619, atol=1e-6)
    has_filter = queries["search_infm_params_text"].ne("").to_numpy()
    results = {}
    for weight in FILTER_WEIGHTS:
        scores = recall_per_query(predictions[weight], truth)
        results[weight] = scores
        print(
            f"filter_weight={weight:.1f}: overall={scores.mean():.6f}, "
            f"filtered={scores[has_filter].mean():.6f}, "
            f"delta={scores.mean()-baseline.mean():+.6f}",
            flush=True,
        )
    print(
        f"E01: overall={baseline.mean():.6f}, "
        f"filtered={baseline[has_filter].mean():.6f}",
        flush=True,
    )

    selected = max(FILTER_WEIGHTS, key=lambda weight: results[weight].mean())
    improvement = results[selected] - baseline
    rng = np.random.default_rng(42)
    bootstrap = improvement[
        rng.integers(0, len(improvement), size=(5000, len(improvement)))
    ].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    print(
        f"Best weight={selected:.1f}; paired delta={improvement.mean():+.6f}; "
        f"bootstrap 95% interval=[{low:+.6f}, {high:+.6f}]",
        flush=True,
    )
    if not make_submission:
        print("Validation only; submission_02.csv was not created", flush=True)
        return
    if improvement.mean() <= 0:
        print("No local gain; submission_02.csv was not created", flush=True)
        return

    benchmark_queries = pd.read_parquet(DATA / "benchmark_queries.parquet")
    benchmark_items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=ITEM_COLS)
    print(
        f"Benchmark: {len(benchmark_queries)} queries, {len(benchmark_items)} items",
        flush=True,
    )
    benchmark_predictions = retrieve(
        benchmark_queries, benchmark_items, (selected,)
    )[selected]
    answer = pd.DataFrame(
        {
            "query_id": benchmark_queries["query_id"],
            "answer": [" ".join(row) for row in benchmark_predictions],
        }
    )
    valid_ids = set(benchmark_items["item_id"])
    assert answer["query_id"].is_unique
    assert answer["query_id"].str.len().eq(16).all()
    assert all(len(row) == TOP_K and len(set(row)) == TOP_K for row in benchmark_predictions)
    assert all(set(row) <= valid_ids for row in benchmark_predictions)
    output = ROOT / "submissions" / "submission_02.csv"
    answer.to_csv(output, index=False)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-submission", action="store_true")
    main(make_submission=parser.parse_args().make_submission)
