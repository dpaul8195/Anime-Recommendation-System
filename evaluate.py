import gc
import numpy as np
from tqdm import tqdm
import tensorflow as tf


def batched_predict_enc(model, users_enc, animes_enc, batch_size=8192):
    if len(users_enc) == 0:
        return np.array([])
    ds = tf.data.Dataset.from_tensor_slices((users_enc, animes_enc))
    ds = (
        ds.batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .map(
            lambda u, a: (
                {"user": tf.expand_dims(u, -1), "anime": tf.expand_dims(a, -1)}
            )
        )
    )
    preds = model.predict(ds, verbose=0)
    return preds.flatten()


def evaluate_sampled_ranking_safe(
    test_df,
    all_items=None,
    n_negatives=50,
    k=10,
    user_col="user",
    anime_col="anime",
    user_enc_map=None,
    anime_enc_map=None,
    batch_size=1024,
    max_users=None,
    random_state=42,
    verbose=True,
):
    rng = np.random.default_rng(random_state)

    if all_items is None:
        try:
            all_items = test_df[anime_col].unique()
        except Exception:
            if anime_enc_map is not None:
                all_items = np.array(list(anime_enc_map.keys()), dtype=np.int32)
            else:
                raise ValueError("Unable to determine all_items.")
    else:
        all_items = np.array(all_items, dtype=np.int32)

    user_pos = {}
    for _, row in test_df.iterrows():
        u = int(row[user_col])
        a = int(row[anime_col])
        user_pos.setdefault(u, set()).add(a)

    user_list = list(user_pos.keys())
    if max_users is not None:
        user_list = user_list[:max_users]

    precisions, recalls, ndcgs, hits = [], [], [], []
    evaluated = 0

    iterator = tqdm(user_list, desc="Eval users", disable=not verbose)

    for user_raw in iterator:
        pos_set_raw = user_pos[user_raw]
        if user_enc_map is not None:
            u_enc = user_enc_map.get(user_raw)
        else:
            u_enc = user_raw
        if u_enc is None:
            continue

        neg_pool = np.setdiff1d(
            all_items, np.array(list(pos_set_raw)), assume_unique=True
        )
        if neg_pool.size == 0:
            continue

        for pos_raw in pos_set_raw:
            pos_enc = (
                anime_enc_map.get(pos_raw) if anime_enc_map is not None else pos_raw
            )
            if pos_enc is None:
                continue

            sample_size = min(n_negatives, len(neg_pool))
            negs = rng.choice(neg_pool, size=sample_size, replace=False)
            candidates_raw = np.concatenate(([pos_raw], negs)).astype(np.int32)

            enc_candidates = []
            valid_raw = []
            for r in candidates_raw:
                enc = anime_enc_map.get(int(r)) if anime_enc_map is not None else int(r)
                if enc is None:
                    continue
                enc_candidates.append(enc)
                valid_raw.append(int(r))
            if not enc_candidates:
                continue

            users_arr = np.array([u_enc] * len(enc_candidates), dtype=np.int32)
            animes_arr = np.array(enc_candidates, dtype=np.int32)

            scores = batched_predict_enc(
                model, users_arr, animes_arr, batch_size=batch_size
            )
            if scores.size == 0:
                continue

            k2 = min(k, len(scores))
            topk_idx = np.argpartition(-scores, k2 - 1)[:k2]
            topk_sorted = topk_idx[np.argsort(-scores[topk_idx])]
            topk_raw = np.array(valid_raw)[topk_sorted]

            hits.append(1.0 if pos_raw in topk_raw else 0.0)
            precisions.append(np.sum([1 for x in topk_raw[:k] if x in pos_set_raw]) / k)
            recalls.append(
                np.sum([1 for x in topk_raw[:k] if x in pos_set_raw])
                / (len(pos_set_raw) + 1e-12)
            )

            dcg = 0.0
            for i, it in enumerate(topk_raw[:k]):
                rel = 1.0 if it in pos_set_raw else 0.0
                dcg += (2**rel - 1) / np.log2(i + 2)
            idcg = sum(
                (2**1 - 1) / np.log2(i + 2) for i in range(min(len(pos_set_raw), k))
            )
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

            evaluated += 1

        if evaluated % 1000 == 0:
            gc.collect()

    metrics = {
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"HitRate@{k}": float(np.mean(hits)) if hits else 0.0,
        "evaluated_pairs": int(evaluated),
    }
    return metrics


sampled_metrics = evaluate_sampled_ranking_safe(
    test_df,
    all_items=df.anime_id.unique() if "df" in globals() else None,
    n_negatives=50,
    k=10,
    user_col="user",
    anime_col="anime",
    user_enc_map=user2user_encoded,
    anime_enc_map=anime2anime_encoded,
    batch_size=1024,
    max_users=2000,
    random_state=42,
    verbose=True,
)

print("Sampled ranking metrics (safe run):", sampled_metrics)
