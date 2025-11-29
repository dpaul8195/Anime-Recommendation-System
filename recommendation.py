import numpy as np
import pandas as pd
from collections import defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tensorflow as tf

INPUT_DIR = "/kaggle/input/anime-recommendation-database-2020"
EPS = 1e-9

_anime_meta = pd.read_csv(f"{INPUT_DIR}/anime.csv", low_memory=False)
_anime_meta = _anime_meta.replace("Unknown", np.nan)

if "MAL_ID" in _anime_meta.columns:
    _anime_meta["anime_id"] = _anime_meta["MAL_ID"]

if "English name" in _anime_meta.columns:
    _anime_meta["eng_version"] = _anime_meta["English name"]
else:
    _anime_meta["eng_version"] = _anime_meta.get(
        "eng_version", _anime_meta.get("Name", np.nan)
    )

_meta_cols = [
    "anime_id",
    "eng_version",
    "Score",
    "Genres",
    "Episodes",
    "Type",
    "Premiered",
    "Members",
]
meta_df = _anime_meta.loc[:, [c for c in _meta_cols if c in _anime_meta.columns]].copy()

MALID_to_eng = dict(zip(meta_df["anime_id"], meta_df["eng_version"]))
MALID_to_genres = dict(zip(meta_df["anime_id"], meta_df["Genres"]))
MALID_to_score = dict(
    zip(meta_df["anime_id"], meta_df.get("Score", pd.Series([np.nan] * len(meta_df))))
)

try:
    syn_df = pd.read_csv(
        f"{INPUT_DIR}/anime_with_synopsis.csv",
        usecols=["MAL_ID", "Name", "sypnopsis"],
        low_memory=False,
    )
except Exception:
    syn_df = pd.read_csv(f"{INPUT_DIR}/anime_with_synopsis.csv", low_memory=False)
    if "MAL_ID" not in syn_df.columns:
        possible_id = [
            c for c in syn_df.columns if "mal" in c.lower() or "id" in c.lower()
        ]
        if possible_id:
            syn_df = syn_df.rename(columns={possible_id[0]: "MAL_ID"})
    if "sypnopsis" not in syn_df.columns:
        if "synopsis" in syn_df.columns:
            syn_df = syn_df.rename(columns={"synopsis": "sypnopsis"})

SYN_by_id = {}
if "MAL_ID" in syn_df.columns and "sypnopsis" in syn_df.columns:
    SYN_by_id = dict(zip(syn_df["MAL_ID"], syn_df["sypnopsis"]))


def extract_weights(name, model):
    layer = model.get_layer(name)
    W = layer.get_weights()[0].astype(np.float32)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return W / norms


def topk_similar_vecs(weights, query_index, top_k=10, include_self=False):
    query = weights[query_index]
    sims = weights.dot(query)
    top_idx = np.argpartition(
        -sims, range(min(top_k + (0 if include_self else 1), len(sims)))
    )[: top_k + (0 if include_self else 1)]
    top_idx_sorted = top_idx[np.argsort(-sims[top_idx])]
    if not include_self:
        top_idx_sorted = top_idx_sorted[top_idx_sorted != query_index]
        top_idx_sorted = top_idx_sorted[:top_k]
    else:
        top_idx_sorted = top_idx_sorted[:top_k]
    return top_idx_sorted, sims


def safe_get_anime_name(malid):
    return MALID_to_eng.get(malid, str(malid))


def safe_get_synopsis(malid):
    return SYN_by_id.get(malid, "Synopsis not available")


def find_similar_animes_by_malid(
    malid,
    n=10,
    model=None,
    anime2anime_encoded=None,
    anime_encoded2anime=None,
    anime_weights=None,
):
    if anime_weights is None:
        anime_weights = extract_weights("anime_embedding", model)

    enc = anime2anime_encoded.get(int(malid))
    if enc is None:
        raise KeyError(f"MAL id {malid} not present in encodings")

    idxs, sims = topk_similar_vecs(anime_weights, enc, top_k=n, include_self=False)

    rows = []
    for i in idxs:
        decoded = anime_encoded2anime.get(int(i))
        rows.append(
            {
                "name": safe_get_anime_name(decoded),
                "similarity": float(sims[i]),
                "genre": MALID_to_genres.get(decoded, ""),
                "sypnopsis": safe_get_synopsis(decoded),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("similarity", ascending=False)
        .reset_index(drop=True)
    )


def batched_predict_for_user(model, user_encoded, candidate_item_encs, batch_size=4096):
    ds_users = np.full((len(candidate_item_encs),), user_encoded, dtype=np.int32)
    ds_items = np.array(candidate_item_encs, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices(
        ((ds_users, ds_items), np.zeros_like(ds_users, dtype=np.float32))
    )
    dataset = dataset.batch(batch_size).map(
        lambda x, y: (
            {"user": tf.expand_dims(x[0], -1), "anime": tf.expand_dims(x[1], -1)},
            y,
        )
    )
    preds = model.predict(dataset, verbose=0)
    return preds.flatten()


def recommend_top_n_for_user(
    raw_user_id,
    model,
    user2user_encoded,
    anime2anime_encoded,
    anime_encoded2anime,
    df_meta=meta_df,
    top_n=10,
    batch_size=4096,
):
    watched = rating_df[rating_df.user_id == raw_user_id].anime_id.unique()

    all_mal_ids = list(anime2anime_encoded.keys())
    candidates = [mid for mid in all_mal_ids if mid not in set(watched)]
    candidate_encs = [
        anime2anime_encoded[mid] for mid in candidates if mid in anime2anime_encoded
    ]

    user_enc = user2user_encoded.get(raw_user_id)
    if user_enc is None:
        raise KeyError(f"user {raw_user_id} not in user2user_encoded")

    scores = batched_predict_for_user(
        model, user_enc, candidate_encs, batch_size=batch_size
    )
    top_idx = np.argpartition(-scores, range(min(top_n, len(scores))))[:top_n]
    top_sorted = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for i in top_sorted:
        malid = anime_encoded2anime[candidate_encs[i]]
        results.append(
            {
                "name": safe_get_anime_name(malid),
                "pred_rating": float(scores[i]),
                "genre": MALID_to_genres.get(malid, ""),
                "sypnopsis": safe_get_synopsis(malid),
            }
        )

    df_res = pd.DataFrame(results).reset_index(drop=True)
    df_res.index = df_res.index + 1
    return df_res


def getFavGenre(frame, plot=False):
    all_genres = defaultdict(int)
    genres_list = []
    for genres in frame.get("Genres", frame.get("genre", [])):
        if isinstance(genres, str):
            for genre in genres.split(","):
                g = genre.strip()
                genres_list.append(g)
                all_genres[g] += 1

    if plot:
        wc = WordCloud(
            width=700, height=400, background_color="white"
        ).generate_from_frequencies(all_genres)
        plt.figure(figsize=(10, 8))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    return genres_list


def get_user_preferences(user_id, rating_df, df_meta, plot=False, verbose=0):
    animes_by_user = rating_df[rating_df.user_id == user_id]
    if len(animes_by_user) == 0:
        return pd.DataFrame([], columns=["eng_version", "Genres"])

    user_rating_percentile = np.percentile(animes_by_user.rating, 75)
    top_user = animes_by_user[animes_by_user.rating >= user_rating_percentile]
    top_ids = top_user.anime_id.unique()

    anime_rows = df_meta[df_meta["anime_id"].isin(top_ids)][["eng_version", "Genres"]]

    if verbose:
        print(
            f"> User #{user_id}: {len(animes_by_user)} rated, avg top-rating = {top_user.rating.mean():.3f}"
        )

    if plot:
        getFavGenre(anime_rows, plot=True)

    return anime_rows
