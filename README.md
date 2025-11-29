# Anime Recommendation System — Reproducible Instructions

- Project: Anime Recommendation System — Using Collaborative Filtering
- Author: Debabrata Paul (Roll: 242123009) — Dept. of Mathematics, IIT Guwahati
- Dataset: Anime Recommendation Database 2020 (MyAnimeList)

# Quick summary

This repository contains a complete pipeline to reproduce the project:

- Preprocessing — load animelist.csv, filter active users, normalize ratings, encode IDs, produce train_df / test_df. 

- Model training — neural collaborative filtering with embedding layers, training, checkpointing.

- Extract embeddings & generate recommendations — load trained model, normalize embeddings, find similar items/users, produce top-N for a user. 
- Evaluation — memory-safe sampled ranking evaluation (Precision@K, Recall@K, NDCG, HitRate). (evaluate.py or evaluate.ipynb)


# Repository structure
```
anime-recsys/
├─ data/                        
├─ scripts/
│  ├─ preprocessing.py
│  ├─ model.py
│  ├─ recommendation.py
│  └─ evaluate.py            
├─ requirements.txt            
└─ README.md
```

# Environment and dependencies

We recommend using a conda environment (or virtualenv). Minimal packages:

## requirements.txt
```
numpy==1.25.0
pandas==2.2.0
tensorflow==2.12.0
scikit-learn==1.2.2
tqdm==4.65.0
matplotlib==3.8.0
wordcloud==1.9.2
ipython==8.21.0
jupyterlab==4.0.0
```

## Install:

conda env create -f environment.yml
conda activate anime-recsys
    or with pip in a venv:
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt


# GPU notes:

Use a CUDA-capable machine if you want faster training. The code uses TensorFlow MirroredStrategy optionally — ensure tensorflow GPU build and appropriate CUDA/cuDNN are installed.

If GPU not available, training will run on CPU (slower). Reduce batch sizes then.

# config / constants

## Create config.yaml (or edit variables at top of scripts):
```
INPUT_DIR: "/kaggle/input/anime-recommendation-database-2020"
OUTPUT_DIR: "./outputs"
MIN_RATINGS_PER_USER: 400
TEST_SIZE_PER_USER: 0.01
EMBEDDING_SIZE: 128
BATCH_SIZE: 10000
EPOCHS: 20
LR_START: 1e-5
LR_MAX: 5e-5
LR_MIN: 1e-5
RAMPUP_EPOCHS: 5
SEED: 42
SAFE_EVAL_MAX_USERS: 2000
SAFE_EVAL_NEGATIVES: 50
SAFE_EVAL_BATCH_SIZE: 1024
```

Load with pyyaml or simply copy into a cell at top of notebook as variables.

# Reproducible run: notebooks

If your code is already in notebooks, run them in this order:

## preprocessing.py

```
Loads dataset from INPUT_DIR.

Filters users with >= 400 ratings.

Normalizes ratings to [0,1].

Encodes user and anime to contiguous indices.

Produces train_df, test_df, saves CSVs if you want.

Save mapping dicts (user2user_encoded, anime2anime_encoded) via pickle if you plan to run scripts separately.

```
## model.py

```
Loads train_df and test_df from preprocessing (or uses variables in memory).

Builds RecommenderNet() model (embedding dims set via config).

Trains with callbacks (ModelCheckpoint -> weights.h5), early stopping.

Saves final model as anime_model.h5.

```
## recommendation.py

```
Loads anime_model.h5.

Extracts normalized anime_weights and user_weights.

Functions: find_similar_animes, find_similar_users, get_recommended_animes.

Saves Results and optionally top-n CSV.

```
## evaluate.py

```
This script performs sampled ranking evaluation using a memory-safe method:

Computes:

- Precision@10

- Recall@10

- NDCG@10

- HitRate@10

Uses only a subset of users (max_users=2000) to prevent OOM

Prints or saves final metrics

```

# Example run_all.sh
```
#!/usr/bin/env bash
set -e
INPUT_DIR="/kaggle/input/anime-recommendation-database-2020"
OUT="./outputs"
mkdir -p $OUT

python scripts/preprocessing.py --input_dir $INPUT_DIR --out_dir $OUT
python scripts/model.py --train_csv $OUT/train_df.csv --val_csv $OUT/test_df.csv --weights_out $OUT/weights.h5
python scripts/recommendation.py --model $OUT/anime_model.h5 --mappings $OUT/mappings.pkl --user 52448 --top_k 10 --out $OUT/recs_user_52448.csv
python scripts/evaluate.py --model $OUT/anime_model.h5 --test_csv $OUT/test_df.csv --mappings $OUT/mappings.pkl --max_users 2000
```

