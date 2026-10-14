"""
Movie Recommendation System - model builder
-------------------------------------------
Pipeline:
  1. Load raw dataset (CSV)
  2. Clean data (duplicates, whitespace, case, missing values)
  3. Feature engineering -> one "tags" text per movie
  4. Text vectorization (TF-IDF)
  5. Cosine similarity between every pair of movies
  6. Save top-N recommendations (+ reasons) for the web app

Run:  python ml/build_model.py
Output: data/movies.js  (loaded by index.html)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "ml" / "movies_raw.csv"
OUT_JS = ROOT / "data" / "movies.js"
TOP_N = 10


# ---------------------------------------------------------------- 1. LOAD
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


# --------------------------------------------------------------- 2. CLEAN
def split_list(value: str) -> list[str]:
    """'Comedy|Drama' -> ['Comedy', 'Drama']"""
    if not isinstance(value, str) or not value.strip():
        return []
    return [v.strip() for v in value.split("|") if v.strip()]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # strip stray whitespace in every text column
    text_cols = ["title", "language", "genres", "director", "cast", "keywords", "wiki_title", "overview"]
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip().astype(object)

    # remove duplicate movies
    before = len(df)
    df = df.drop_duplicates(subset=["title", "year"]).reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate row(s)")

    # missing values
    print("Missing values before cleaning:\n", df.isna().sum()[df.isna().sum() > 0], sep="")
    for col in ["keywords", "director", "cast", "overview"]:
        df[col] = df[col].fillna("")
    df["imdb_rating"] = df["imdb_rating"].fillna(df["imdb_rating"].median())

    # normalise genre casing:  'DRAMA' / 'drama' -> 'Drama'
    df["genres"] = df["genres"].apply(lambda g: "|".join(x.title() if x.lower() != "sci-fi" else "Sci-Fi"
                                                        for x in split_list(g)))

    # list columns used later
    for col in ["genres", "director", "cast", "keywords"]:
        df[col + "_list"] = df[col].apply(split_list)

    df["year"] = df["year"].astype(int)
    df["id"] = range(len(df))
    return df


# --------------------------------------------------- 3. FEATURE ENGINEERING
def collapse(name: str) -> str:
    """'Aamir Khan' -> 'aamirkhan' so 'Khan' alone doesn't match different people."""
    return name.lower().replace(" ", "").replace(".", "").replace("-", "")


def build_tags(row: pd.Series) -> str:
    genres = [g.lower().replace("-", "") for g in row["genres_list"]]
    director = [collapse(d) for d in row["director_list"]]
    cast = [collapse(c) for c in row["cast_list"][:3]]            # top-3 billed actors
    keywords = [k.lower().replace(" ", "") for k in row["keywords_list"]]
    language = [row["language"].lower()]

    # repeating a field = giving it more weight in the vector
    parts = (genres * 2) + (director * 2) + cast + (keywords * 2) + language + [row["overview"].lower()]
    return " ".join(parts)


# --------------------------------------------------- 4 & 5. VECTORS + SIMILARITY
def similarity_matrix(tags: pd.Series, method: str = "tfidf") -> np.ndarray:
    if method == "tfidf":
        vec = TfidfVectorizer(stop_words="english", max_features=5000)
    else:
        vec = CountVectorizer(stop_words="english", max_features=5000)
    matrix = vec.fit_transform(tags)                 # shape: (movies, vocabulary)
    print(f"[{method}] vector shape: {matrix.shape}")
    return cosine_similarity(matrix)                  # shape: (movies, movies)


# --------------------------------------------------- 6. RECOMMEND
def recommend(df: pd.DataFrame, sim: np.ndarray, title: str, n: int = TOP_N) -> pd.DataFrame:
    """Return the n most similar movies to `title`."""
    idx = df.index[df["title"].str.lower() == title.lower()][0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:n]
    out = df.iloc[[i for i, _ in scores]][["title", "year", "genres"]].copy()
    out["similarity"] = [round(s, 3) for _, s in scores]
    return out


def reasons(a: pd.Series, b: pd.Series) -> list[str]:
    """Human-readable 'why' for a recommendation (explainability)."""
    why = []
    d = set(a["director_list"]) & set(b["director_list"])
    if d:
        why.append("Director: " + ", ".join(sorted(d)))
    c = [x for x in b["cast_list"] if x in a["cast_list"]]
    if c:
        why.append("Cast: " + ", ".join(c[:2]))
    g = [x for x in b["genres_list"] if x in a["genres_list"]]
    if g:
        why.append("Genre: " + ", ".join(g))
    k = [x for x in b["keywords_list"] if x in a["keywords_list"]]
    if k:
        why.append("Theme: " + ", ".join(k[:2]))
    return why


# --------------------------------------------------- EVALUATION
def genre_precision_at_k(df: pd.DataFrame, sim: np.ndarray, k: int = TOP_N) -> float:
    """% of recommended movies that share at least one genre with the selected movie."""
    hits = total = 0
    for i in range(len(df)):
        top = np.argsort(sim[i])[::-1]
        top = [j for j in top if j != i][:k]
        g = set(df.at[i, "genres_list"])
        hits += sum(bool(g & set(df.at[j, "genres_list"])) for j in top)
        total += k
    return round(100 * hits / total, 1)


# --------------------------------------------------- MAIN
def main() -> None:
    df = clean_data(load_data(RAW_CSV))
    df["tags"] = df.apply(build_tags, axis=1)

    sim = similarity_matrix(df["tags"], "tfidf")
    sim_count = similarity_matrix(df["tags"], "count")

    metrics = {
        "tfidf_precision": genre_precision_at_k(df, sim),
        "count_precision": genre_precision_at_k(df, sim_count),
        "movies": len(df),
        "top_n": TOP_N,
    }
    print(f"Genre Precision@{TOP_N}  TF-IDF: {metrics['tfidf_precision']}%  "
          f"Count: {metrics['count_precision']}%")

    print("\nSample - movies similar to '3 Idiots':")
    print(recommend(df, sim, "3 Idiots", 5).to_string(index=False))

    # ---- export for the web app
    movies = []
    for _, r in df.iterrows():
        movies.append({
            "id": int(r["id"]),
            "title": r["title"],
            "year": int(r["year"]),
            "language": r["language"],
            "genres": r["genres_list"],
            "director": r["director_list"],
            "cast": r["cast_list"],
            "rating": float(r["imdb_rating"]),
            "overview": r["overview"],
            "wiki": r["wiki_title"],
        })

    recs = {}
    for i in range(len(df)):
        order = [j for j in np.argsort(sim[i])[::-1] if j != i][:TOP_N]
        recs[i] = [[int(j), round(float(sim[i][j]), 4), reasons(df.iloc[i], df.iloc[j])] for j in order]

    stats = {
        "genres": df["genres_list"].explode().value_counts().head(10).to_dict(),
        "languages": df["language"].value_counts().to_dict(),
        "decades": df["year"].floordiv(10).mul(10).astype(str).add("s").value_counts().sort_index().to_dict(),
        "avg_rating": round(float(df["imdb_rating"].mean()), 2),
    }

    payload = {"movies": movies, "recs": recs, "stats": stats, "metrics": metrics}
    OUT_JS.parent.mkdir(exist_ok=True)
    OUT_JS.write_text("window.MOVIE_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n",
                      encoding="utf-8")
    print(f"\nSaved {OUT_JS.relative_to(ROOT)}  ({OUT_JS.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
