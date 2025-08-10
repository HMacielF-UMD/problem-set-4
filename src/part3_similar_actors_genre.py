'''
PART 2: SIMILAR ACTROS BY GENRE

Using the imbd_movies dataset:
- Create a data frame, where each row corresponds to an actor, each column represents a genre, and each cell captures how many times that row's actor has appeared in that column’s genre 
- Using this data frame as your “feature matrix”, select an actor (called your “query”) for whom you want to find the top 10 most similar actors based on the genres in which they’ve starred 
- - As an example, select the row from your data frame associated with Chris Hemsworth, actor ID “nm1165110”, as your “query” actor
- Use sklearn.metrics.DistanceMetric to calculate the euclidean distances between your query actor and all other actors based on their genre appearances
- - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html
- Output a CSV continaing the top ten actors most similar to your query actor using cosine distance 
- - Name it 'similar_actors_genre_{current_datetime}.csv' to `/data`
- - For example, the top 10 for Chris Hemsworth are:  
        nm1165110 Chris Hemsworth
        nm0000129 Tom Cruise
        nm0147147 Henry Cavill
        nm0829032 Ray Stevenson
        nm5899377 Tiger Shroff
        nm1679372 Sudeep
        nm0003244 Jordi Mollà 
        nm0636280 Richard Norton
        nm0607884 Mark Mortimer
        nm2018237 Taylor Kitsch
- Describe in a print() statement how this list changes based on Euclidean distance
- Make sure your code is in line with the standards we're using in this class
'''

import os
import json
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import cosine_distances

# Params
INPUT_PATH = "data/movies.json"   # JSONL, one movie per line, with 'actors' and 'genres'
QUERY_ACTOR_ID = "nm1165110"      # Chris Hemsworth

# Accumulate actor -> genre counts
actor_names = {}
actor_genre_counts = defaultdict(Counter)
genres_set = set()

with open(INPUT_PATH, "r", encoding="utf-8") as in_file:
    for line in in_file:
        if not line.strip():
            continue
        movie = json.loads(line)
        actors = movie.get("actors") or movie.get("casts") or []
        genres = movie.get("genres", [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(",") if g.strip()]
        elif isinstance(genres, list):
            genres = [str(g).strip() for g in genres if str(g).strip()]
        else:
            genres = []
        if not actors or not genres:
            continue
        genres_set.update(genres)
        for aid, aname in actors:
            actor_names[aid] = aname
            for g in genres:
                actor_genre_counts[aid][g] += 1

if not actor_genre_counts:
    raise RuntimeError("No actor-genre data accumulated. Check input format and fields 'actors' and 'genres'.")

# Build dataframe (actor x genre)
genre_cols = sorted(genres_set)
df = pd.DataFrame([
    {"actor_id": aid, "actor_name": actor_names.get(aid, aid), **{g: counts.get(g, 0) for g in genre_cols}}
    for aid, counts in actor_genre_counts.items()
])

# Feature matrix
X = df[genre_cols].to_numpy(float)
if QUERY_ACTOR_ID not in set(df["actor_id"]):
    raise ValueError(f"Query actor {QUERY_ACTOR_ID} not found.")
query_idx = df.index[df["actor_id"] == QUERY_ACTOR_ID][0]
qvec = X[query_idx].reshape(1, -1)

# Cosine distance for similarity
cos_dists = cosine_distances(qvec, X)[0]
cos_order = [i for i in cos_dists.argsort() if i != query_idx][:10]
cos_top10 = df.iloc[cos_order][["actor_id", "actor_name"]].copy()
cos_top10["cosine_distance"] = cos_dists[cos_order]

# Save cosine top 10 to /data
os.makedirs("data", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join("data", f"similar_actors_genre_{timestamp}.csv")
cos_top10.to_csv(out_path, index=False)
print(f"Wrote: {out_path}")

# Euclidean distance for comparison
euc_dists = DistanceMetric.get_metric("euclidean").pairwise(qvec, X)[0]
euc_order = [i for i in euc_dists.argsort() if i != query_idx][:10]
euc_top10 = df.iloc[euc_order][["actor_id", "actor_name"]].copy()
euc_top10["euclidean_distance"] = euc_dists[euc_order]

# Prints
qname = df.loc[query_idx, "actor_name"]
print(f"\nQuery actor: {QUERY_ACTOR_ID} {qname}\n")

print("Top 10 by cosine distance (smaller is more similar):")
for _, row in cos_top10.iterrows():
    print(f"{row['actor_id']} {row['actor_name']} | cosine={row['cosine_distance']:.4f}")

print("\nTop 10 by Euclidean distance (smaller is more similar):")
for _, row in euc_top10.iterrows():
    print(f"{row['actor_id']} {row['actor_name']} | euclidean={row['euclidean_distance']:.4f}")

# Brief description of how the list changes under Euclidean
cos_ids = list(cos_top10["actor_id"])
euc_ids = list(euc_top10["actor_id"])
overlap = [a for a in cos_ids if a in euc_ids]
only_cos = [a for a in cos_ids if a not in euc_ids]
only_euc = [a for a in euc_ids if a not in cos_ids]
print("\nChange with Euclidean vs cosine:")
print(f"- Overlap: {len(overlap)} actors")
print(f"- Only in cosine top 10: {only_cos}")
print(f"- Only in Euclidean top 10: {only_euc}")