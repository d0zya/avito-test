"""Test pseudo-relevance feedback with the fine-tuned E5 encoder.

Uses E04's fixed validation corpus and embeddings. No new training or
benchmark file is produced by this screening experiment.
"""

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from e01_baseline import DATA, ITEM_COLS, QUERY_COLS, ROOT, validation_queries
from e04_e5_finetune import MODEL_OUTPUT, query_text, validation_items


def main() -> None:
    train = pd.read_parquet(DATA / "train.parquet", columns=QUERY_COLS + ITEM_COLS)
    queries, truth = validation_queries(train)
    items = validation_items(train, truth)
    del train
    vectors = np.load(ROOT / "artifacts" / "e04_validation_items_50000.npy")
    assert vectors.shape == (len(items), 768)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(str(MODEL_OUTPUT), device=device, local_files_only=True)
    query_vectors = model.encode(
        query_text(queries), batch_size=128, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    item_vectors = torch.as_tensor(vectors.astype(np.float32), device=device)
    query_vectors = torch.as_tensor(query_vectors, device=device)
    item_locations = torch.as_tensor(items["item_location_id"].to_numpy().copy(), device=device)
    query_locations = torch.as_tensor(queries["search_location_id"].to_numpy().copy(), device=device)
    item_ids = items["item_id"].to_numpy()
    variants = [(count, beta, location) for location in (0.0, 0.10) for count in (3, 10) for beta in (0.2, 0.5)]
    hits = {key: [] for key in [(0, 0.0, 0.0), (0, 0.0, 0.10)] + variants}
    with torch.inference_mode():
        for start in range(0, len(queries), 16):
            stop = min(start + 16, len(queries))
            q = query_vectors[start:stop]
            location_match = query_locations[start:stop, None] == item_locations[None, :]
            base_scores = q @ item_vectors.T
            for location in (0.0, 0.10):
                first = base_scores + location * location_match
                top = torch.topk(first, k=50, dim=1).indices
                top_ids = item_ids[top.cpu().numpy()]
                hits[(0, 0.0, location)].extend(
                    len(set(row) & gold) / len(gold) for row, gold in zip(top_ids, truth[start:stop])
                )
                for count in (3, 10):
                    feedback = item_vectors[top[:, :count]].mean(dim=1)
                    for beta in (0.2, 0.5):
                        expanded = torch.nn.functional.normalize((1 - beta) * q + beta * feedback, dim=1)
                        scores = expanded @ item_vectors.T + location * location_match
                        updated = torch.topk(scores, k=50, dim=1).indices
                        updated_ids = item_ids[updated.cpu().numpy()]
                        hits[(count, beta, location)].extend(
                            len(set(row) & gold) / len(gold) for row, gold in zip(updated_ids, truth[start:stop])
                        )
            if stop % 160 == 0 or stop == len(queries):
                print(f"Evaluated {stop}/{len(queries)} queries", flush=True)
    for key, values in hits.items():
        print(f"feedback_top={key[0]}, beta={key[1]:.1f}, location={key[2]:.2f}: Recall@50={np.mean(values):.6f}", flush=True)


if __name__ == "__main__":
    main()
