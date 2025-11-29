

import argparse
import json
import os
import pickle
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- Config / defaults ----------
DEFAULT_INPUT_DIR = "/kaggle/input/anime-recommendation-database-2020"
ARTIFACT_DIR = "artifacts"
TRAIN_OUT = os.path.join(ARTIFACT_DIR, "train_df.parquet")
TEST_OUT = os.path.join(ARTIFACT_DIR, "test_df.parquet")
MAPPINGS_OUT = os.path.join(ARTIFACT_DIR, "mappings.pkl")
STATS_OUT = os.path.join(ARTIFACT_DIR, "dataset_stats.json")
RANDOM_STATE = 42


def ensure_artifact_dir():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_ratings(input_dir: str, nrows: int = None) -> pd.DataFrame:
    path = os.path.join(input_dir, "animelist.csv")
    print(f"[INFO] Loading ratings from: {path}")
    usecols = ["user_id", "anime_id", "rating"]
    # low_memory=False to avoid dtype warnings; set nrows for quick tests if needed
    df = pd.read_csv(path, usecols=usecols, low_memory=False, nrows=nrows)
    print(f"[INFO] Raw interactions: {len(df):,}")
    return df


def filter_active_users(df: pd.DataFrame, min_ratings: int = 400) -> pd.DataFrame:
    print(f"[INFO] Filtering users with at least {min_ratings} ratings...")
    counts = df["user_id"].value_counts()
    active_users = counts[counts >= min_ratings].index
    df_filtered = df[df["user_id"].isin(active_users)].copy()
    print(f"[INFO] Interactions after filtering: {len(df_filtered):,}")
    return df_filtered


def normalize_ratings(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    # Assume rating numeric (e.g. 1..10). Keep original min/max for later denormalization if needed.
    rmin = float(df["rating"].min())
    rmax = float(df["rating"].max())
    if rmax == rmin:
        raise ValueError("All ratings identical; cannot min-max normalize.")
    df = df.copy()
    df["rating"] = df["rating"].astype(float)
    df["rating"] = (df["rating"] - rmin) / (rmax - rmin)
    print(f"[INFO] Ratings normalized to [0,1] using min={rmin}, max={rmax}")
    return df, rmin, rmax


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    dup = df.duplicated().sum()
    if dup > 0:
        print(f"[INFO] Removing {dup} duplicated rows")
        df = df.drop_duplicates().copy()
    else:
        print("[INFO] No duplicate rows found")
    return df


def encode_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Adds columns 'user' and 'anime' (encoded contiguous indices), and returns mapping dicts.
    Returns: df_encoded, mappings (dict)
    """
    print("[INFO] Encoding user and anime IDs to contiguous indices...")
    user_ids = df["user_id"].unique()
    anime_ids = df["anime_id"].unique()

    user2user_encoded = {int(x): int(i) for i, x in enumerate(user_ids)}
    user_encoded2user = {int(i): int(x) for i, x in enumerate(user_ids)}

    anime2anime_encoded = {int(x): int(i) for i, x in enumerate(anime_ids)}
    anime_encoded2anime = {int(i): int(x) for i, x in enumerate(anime_ids)}

    df = df.copy()
    df["user"] = df["user_id"].map(user2user_encoded).astype("int32")
    df["anime"] = df["anime_id"].map(anime2anime_encoded).astype("int32")

    mappings = {
        "user2user_encoded": user2user_encoded,
        "user_encoded2user": user_encoded2user,
        "anime2anime_encoded": anime2anime_encoded,
        "anime_encoded2anime": anime_encoded2anime,
    }

    print(
        f"[INFO] Num users: {len(user2user_encoded):,}, Num animes: {len(anime2anime_encoded):,}"
    )
    return df, mappings


def user_wise_train_test_split(
    df: pd.DataFrame, test_size_per_user: float = 0.01, seed: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs user-wise train/test split: for each user, sample test_size_per_user fraction.
    Returns train_df, test_df
    """
    print(
        f"[INFO] Creating user-wise train/test split (test fraction per user = {test_size_per_user})"
    )
    train_list = []
    test_list = []

    # groupby on raw user_id retains original distribution; we already encoded but either works
    grouped = df.groupby("user_id")
    for uid, group in grouped:
        # if user has very few ratings (shouldn't after filtering) ensure at least one in train
        if len(group) <= 1:
            train_list.append(group)
            continue
        train_g, test_g = train_test_split(
            group, test_size=test_size_per_user, random_state=seed, shuffle=True
        )
        train_list.append(train_g)
        test_list.append(test_g)

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    print(f"[INFO] Train size: {len(train_df):,}, Test size: {len(test_df):,}")
    return train_df, test_df


def save_artifacts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mappings: dict,
    rmin: float,
    rmax: float,
):
    ensure_artifact_dir()
    print(f"[INFO] Writing train -> {TRAIN_OUT}")
    train_df.to_parquet(TRAIN_OUT, index=False)
    print(f"[INFO] Writing test -> {TEST_OUT}")
    test_df.to_parquet(TEST_OUT, index=False)

    # Save mappings and rating scaling info
    meta = {
        "mappings": mappings,
        "rating_min": rmin,
        "rating_max": rmax,
        "n_users": len(mappings["user2user_encoded"]),
        "n_animes": len(mappings["anime2anime_encoded"]),
    }
    print(f"[INFO] Writing mappings -> {MAPPINGS_OUT}")
    with open(MAPPINGS_OUT, "wb") as f:
        pickle.dump(meta, f)

    stats = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "rating_min": rmin,
        "rating_max": rmax,
        "n_users": meta["n_users"],
        "n_animes": meta["n_animes"],
    }
    with open(STATS_OUT, "w") as f:
        json.dump(stats, f, indent=2)
    print("[INFO] Artifacts saved.")


def prepare(
    input_dir: str = DEFAULT_INPUT_DIR,
    min_ratings: int = 400,
    test_fraction_per_user: float = 0.01,
    nrows: int = None,
):
    """
    Full preprocessing pipeline. Returns (train_df, test_df, mappings)
    """
    ensure_artifact_dir()
    raw = load_ratings(input_dir=input_dir, nrows=nrows)
    filtered = filter_active_users(raw, min_ratings=min_ratings)
    filtered = remove_duplicates(filtered)
    normalized, rmin, rmax = normalize_ratings(filtered)
    encoded_df, mappings = encode_ids(normalized)
    train_df, test_df = user_wise_train_test_split(
        encoded_df, test_size_per_user=test_fraction_per_user, seed=RANDOM_STATE
    )

    # Save artifacts for later stages (training / evaluation)
    save_artifacts(train_df, test_df, mappings, rmin, rmax)
    return train_df, test_df, mappings


# ----- CLI entrypoint -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Anime Recommendation data")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Folder with animelist.csv",
    )
    parser.add_argument(
        "--min_ratings", type=int, default=400, help="Minimum ratings per user to keep"
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.01,
        help="Fraction per-user to place in test set",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Read only first N rows (for quick debugging)",
    )

    args = parser.parse_args()
    print("[INFO] Starting preprocessing with args:", args)
    train_df, test_df, mappings = prepare(
        input_dir=args.input_dir,
        min_ratings=args.min_ratings,
        test_fraction_per_user=args.test_fraction,
        nrows=args.nrows,
    )
    print("[INFO] Done.")
