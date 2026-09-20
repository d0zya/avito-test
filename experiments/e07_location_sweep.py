"""Check stronger location weights and prepare the third submission."""

import argparse

import numpy as np
import pandas as pd

from e01_baseline import DATA, ITEM_COLS, QUERY_COLS, ROOT, TOP_K, validation_queries
from e06_location import retrieve


WEIGHTS = (4.0, 6.0, 8.0, 12.0, 16.0)
SELECTED_WEIGHT = 12.0


def second_holdout(train: pd.DataFrame, excluded: set[str]) -> tuple[pd.DataFrame, list[set[str]]]:
    contexts = train[QUERY_COLS].drop_duplicates()
    contexts = contexts.loc[
        contexts["search_category"].eq(114)
        & contexts["search_is_delivery_search"].eq(0)
        & ~contexts["search_query"].isin(excluded)
    ]
    empty = contexts.loc[contexts["search_infm_params_text"].eq("")]
    empty = empty.drop_duplicates("search_query").sample(n=630, random_state=1042)
    filled = contexts.loc[
        contexts["search_infm_params_text"].ne("")
        & ~contexts["search_query"].isin(empty["search_query"])
    ]
    filled = filled.drop_duplicates("search_query").sample(n=370, random_state=1043)
    queries = pd.concat([empty, filled], ignore_index=True)
    positives = train[QUERY_COLS + ["item_id"]].drop_duplicates().merge(
        queries.reset_index(names="validation_id"), on=QUERY_COLS, how="inner", sort=False
    )
    truth = positives.groupby("validation_id")["item_id"].agg(lambda x: set(x)).reindex(
        range(len(queries))
    )
    return queries, truth.tolist()


def main(make_submission: bool) -> None:
    train = pd.read_parquet(DATA / "train.parquet", columns=QUERY_COLS + ITEM_COLS)
    first_queries, first_truth = validation_queries(train)
    second_queries, second_truth = second_holdout(train, set(first_queries["search_query"]))
    queries = pd.concat([first_queries, second_queries], ignore_index=True)
    truth = first_truth + second_truth
    items = train[ITEM_COLS].drop_duplicates("item_id").reset_index(drop=True)
    del train
    predictions = retrieve(queries, items, WEIGHTS)
    item_locations = items.set_index("item_id")["item_location_id"].to_dict()
    has_local_positive = np.array([
        any(item_locations[item_id] == location for item_id in relevant)
        for location, relevant in zip(queries["search_location_id"], truth)
    ])
    print(f"Has exact-location positive: {has_local_positive.mean():.3f}", flush=True)
    for weight, rows in predictions.items():
        values = np.array([
            len(set(row) & relevant) / len(relevant)
            for row, relevant in zip(rows, truth)
        ])
        print(
            f"weight={weight:.1f}: first={values[:1000].mean():.6f}, "
            f"second={values[1000:].mean():.6f}, "
            f"local_positive={values[has_local_positive].mean():.6f}, "
            f"no_local_positive={values[~has_local_positive].mean():.6f}",
            flush=True,
        )
    if not make_submission:
        return

    benchmark_queries = pd.read_parquet(DATA / "benchmark_queries.parquet")
    benchmark_items = pd.read_parquet(DATA / "benchmark_items.parquet", columns=ITEM_COLS)
    rows = retrieve(benchmark_queries, benchmark_items, (SELECTED_WEIGHT,))[SELECTED_WEIGHT]
    valid_ids = set(benchmark_items["item_id"])
    assert all(len(row) == TOP_K and len(set(row)) == TOP_K and set(row) <= valid_ids for row in rows)
    answer = pd.DataFrame({
        "query_id": benchmark_queries["query_id"].astype(str),
        "answer": [" ".join(row) for row in rows],
    })
    assert answer["query_id"].is_unique
    assert answer["query_id"].str.len().eq(16).all()
    output = ROOT / "submissions" / "submission_03.csv"
    answer.to_csv(output, index=False)
    answer.to_csv(ROOT / "answer.csv", index=False)
    print(f"Saved {output} and answer.csv", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-submission", action="store_true")
    main(parser.parse_args().make_submission)
