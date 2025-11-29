

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    EarlyStopping,
)
from sklearn.utils import shuffle as sk_shuffle


# ---------------------
# Helpers / Model
# ---------------------
def build_ncf_model(n_users, n_items, embedding_dim=128):
    """
    Simple neural collaborative model using L2-normalized embeddings + dense output.
    Predicts a value in [0,1] using sigmoid and is trained with BCE (consistent with your pipeline).
    """
    user_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name="user")
    item_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name="anime")

    user_emb = layers.Embedding(
        input_dim=n_users,
        output_dim=embedding_dim,
        name="user_embedding",
        embeddings_regularizer=None,
    )(
        user_input
    )  # shape (batch,1,d)
    item_emb = layers.Embedding(
        input_dim=n_items,
        output_dim=embedding_dim,
        name="anime_embedding",
        embeddings_regularizer=None,
    )(item_input)

    # L2-normalize along embedding dim
    user_norm = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(tf.squeeze(x, axis=1), axis=1)
    )(user_emb)
    item_norm = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(tf.squeeze(x, axis=1), axis=1)
    )(item_emb)

    # cosine similarity = dot of normalized embeddings
    dot = tf.keras.layers.Dot(axes=1, normalize=False)(
        [user_norm, item_norm]
    )  # shape (batch, )
    # optionally scale and pass through a small MLP / dense
    x = tf.keras.layers.Reshape((1,))(dot)
    x = tf.keras.layers.Dense(1, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    output = tf.keras.layers.Activation("sigmoid")(x)

    model = Model(inputs=[user_input, item_input], outputs=output)
    return model


def make_dataset_from_df(
    df, batch_size=4096, shuffle_buffer=200000, seed=42, repeat=False, training=True
):
    """
    Expects df with encoded columns 'user' and 'anime' and numeric 'rating'.
    Returns tf.data.Dataset yielding ([user_batch, anime_batch], rating_batch).
    """
    users = df["user"].values.astype(np.int32)
    animes = df["anime"].values.astype(np.int32)
    ratings = df["rating"].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices(((users, animes), ratings))
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), shuffle_buffer),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    if repeat and training:
        ds = ds.repeat()
    return ds


# ---------------------
# Learning rate scheduler (same scheme as your original, but safer)
# ---------------------
def make_lrfn(start_lr, max_lr, min_lr, rampup_epochs, sustain_epochs, exp_decay):
    def lrfn(epoch):
        if epoch < rampup_epochs:
            return (max_lr - start_lr) / float(max(1, rampup_epochs)) * epoch + start_lr
        elif epoch < rampup_epochs + sustain_epochs:
            return max_lr
        else:
            return (max_lr - min_lr) * (
                exp_decay ** (epoch - rampup_epochs - sustain_epochs)
            ) + min_lr

    return lrfn


# ---------------------
# CLI / main
# ---------------------
def main(args):
    artifacts_dir = Path(args.artifacts_dir)
    train_path = artifacts_dir / "train_df.parquet"
    test_path = artifacts_dir / "test_df.parquet"
    mappings_path = artifacts_dir / "mappings.pkl"

    if not train_path.exists() or not test_path.exists() or not mappings_path.exists():
        raise FileNotFoundError(
            f"Missing artifacts. Ensure {train_path}, {test_path}, and {mappings_path} exist."
        )

    print("[INFO] Loading artifacts...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    with open(mappings_path, "rb") as f:
        meta = pickle.load(f)

    n_users = int(meta["n_users"])
    n_items = int(meta["n_animes"])
    print(
        f"[INFO] n_users={n_users:,}, n_items={n_items:,}, train_rows={len(train_df):,}, test_rows={len(test_df):,}"
    )

    # Optionally subsample for debugging to avoid OOM:
    if args.debug and args.subsample_rows is not None:
        print(
            f"[INFO] Debug mode: subsampling {args.subsample_rows} rows from train and test (if available)"
        )
        train_df = train_df.sample(
            min(len(train_df), args.subsample_rows), random_state=42
        )
        test_df = test_df.sample(
            min(len(test_df), args.subsample_rows // 10), random_state=42
        )

    # Build datasets
    batch_size = int(args.batch_size)
    train_ds = make_dataset_from_df(train_df, batch_size=batch_size, training=True)
    val_ds = make_dataset_from_df(test_df, batch_size=batch_size, training=False)

    # Distribution strategy: use mirrored if GPUs available
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        strategy = tf.distribute.MirroredStrategy()
        print(
            f"[INFO] Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas (GPUs)."
        )
    else:
        strategy = tf.distribute.get_strategy()
        print("[INFO] No GPUs found. Using default strategy (CPU).")

    with strategy.scope():
        model = build_ncf_model(
            n_users=n_users, n_items=n_items, embedding_dim=args.embedding_dim
        )
        # compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.start_lr)
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["mae", "mse"]
        )

    model.summary()

    # Callbacks
    lrfn = make_lrfn(
        args.start_lr,
        args.max_lr,
        args.min_lr,
        args.rampup_epochs,
        args.sustain_epochs,
        args.exp_decay,
    )
    lr_callback = LearningRateScheduler(lrfn)
    chkpt_path = args.checkpoint or "weights_best.h5"
    checkpoint = ModelCheckpoint(
        chkpt_path, monitor="val_loss", save_best_only=True, save_weights_only=True
    )
    earlystop = EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    )

    # Fit
    steps_per_epoch = None
    validation_steps = None
    # If dataset size is massive and you used repeat in dataset then you can set steps_per_epoch explicitly.
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint, lr_callback, earlystop],
        verbose=1,
    )

    print("[INFO] Training finished. Best weights saved to:", chkpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Anime Recommendation NCF model")
    parser.add_argument("--artifacts_dir", type=str, default="./artifacts")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--start_lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--rampup_epochs", type=int, default=5)
    parser.add_argument("--sustain_epochs", type=int, default=0)
    parser.add_argument("--exp_decay", type=float, default=0.8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--checkpoint", type=str, default="weights_best.h5")
    parser.add_argument(
        "--debug", action="store_true", help="Subsample data for quick debug run"
    )
    parser.add_argument(
        "--subsample_rows", type=int, default=200000, help="Rows to use in debug mode"
    )
    args = parser.parse_args()
    main(args)
