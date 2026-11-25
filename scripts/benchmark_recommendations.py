"""
Benchmark recommendation models with Course-Enrolled Candidate Filtering.
Compares:
1. Uniform Random Baseline
2. Popularity Baseline
3. Course-Filtered LightGCN Graph Embeddings
4. Production Hybrid Recommendation Engine
Metrics: Precision@K, Recall@K, NDCG@K for K in [5, 10, 20].
"""

import math

import numpy as np
import pandas as pd

from ml.recommendation.hybrid_engine import HybridRecommendationEngine

# Set seeds
np.random.seed(42)

# Load data
df_vle = pd.read_csv("data/raw/vle.csv")
df_student_vle = pd.read_csv("data/raw/studentVle.csv")
df_student_info = pd.read_csv("data/raw/studentInfo.csv")
test_df = pd.read_csv("data/processed/test.csv")

# Load graph embeddings
embeddings = dict(np.load("artifacts/models/graph_embeddings.npz"))

# Split student interactions into train (80%) and holdout test (20%)
unique_pairs = df_student_vle[["id_student", "id_site"]].drop_duplicates()
np_pairs = unique_pairs.to_numpy().copy()
np.random.shuffle(np_pairs)
split_idx = int(len(np_pairs) * 0.8)
train_pairs = set(map(tuple, np_pairs[:split_idx]))
test_pairs = np_pairs[split_idx:]

test_by_student = {}
for s, r in test_pairs:
    test_by_student.setdefault(s, set()).add(r)

# Precompute module resource mapping
module_resources = df_vle.groupby("code_module")["id_site"].unique().to_dict()

# Precompute popularity within each module
pop_counts = df_student_vle.groupby("id_site")["id_student"].nunique().to_dict()

hybrid_engine = HybridRecommendationEngine()

k_list = [5, 10, 20]
methods = [
    "Random (Course-Filtered)",
    "Popularity (Course-Filtered)",
    "LightGCN (Course-Filtered)",
    "Hybrid Engine (Course-Filtered)",
]
results = {m: {f"P@{k}": [] for k in k_list} for m in methods}
for m in methods:
    for k in k_list:
        results[m][f"R@{k}"] = []
        results[m][f"NDCG@{k}"] = []

sampled_students = [s for s in test_by_student if f"stud_{s}" in embeddings][:300]

for s in sampled_students:
    true_items = test_by_student[s]
    stud_modules = df_student_info[df_student_info["id_student"] == s]["code_module"].unique()
    if len(stud_modules) == 0:
        continue

    # Candidate pool restricted to enrolled course(s)
    candidates = []
    for mod in stud_modules:
        candidates.extend(module_resources.get(mod, []))
    candidates = list(set(candidates))
    if len(candidates) < 10:
        continue

    # Filter true items to candidates in enrolled courses
    true_in_course = true_items.intersection(set(candidates))
    if len(true_in_course) == 0:
        continue

    # 1. Random baseline
    rand_ranked = list(np.random.permutation(candidates))

    # 2. Popularity baseline
    pop_ranked = sorted(candidates, key=lambda x: pop_counts.get(x, 0), reverse=True)

    # 3. LightGCN Graph Ranking
    s_emb = embeddings.get(f"stud_{s}")

    def get_graph_score(site_id, s_vec=s_emb):
        r_emb = embeddings.get(f"res_{site_id}")
        if r_emb is not None and s_vec is not None:
            return float(np.dot(s_vec, r_emb))
        return -999.0

    lightgcn_ranked = sorted(candidates, key=get_graph_score, reverse=True)

    # 4. Hybrid Engine
    hybrid_recs = hybrid_engine.recommend_for_student(student_id=s, top_n=20)
    hybrid_ranked = [r["resource_id"] for r in hybrid_recs]
    # Fill remaining from candidates if fewer than 20
    remaining = [c for c in candidates if c not in hybrid_ranked]
    hybrid_ranked.extend(remaining)

    rankings = {
        "Random (Course-Filtered)": rand_ranked,
        "Popularity (Course-Filtered)": pop_ranked,
        "LightGCN (Course-Filtered)": lightgcn_ranked,
        "Hybrid Engine (Course-Filtered)": hybrid_ranked,
    }

    for m_name, ranked_list in rankings.items():
        for k in k_list:
            top_k = ranked_list[:k]
            hits = [1 if item in true_in_course else 0 for item in top_k]
            n_hits = sum(hits)

            p = n_hits / k
            r = n_hits / len(true_in_course)

            dcg = sum(h / math.log2(idx + 2) for idx, h in enumerate(hits))
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(true_in_course))))
            ndcg = (dcg / idcg) if idcg > 0 else 0.0

            results[m_name][f"P@{k}"].append(p)
            results[m_name][f"R@{k}"].append(r)
            results[m_name][f"NDCG@{k}"].append(ndcg)

summary_table = []
for m_name in methods:
    row = {"Method": m_name}
    for k in k_list:
        row[f"P@{k}"] = round(float(np.mean(results[m_name][f"P@{k}"])), 4)
        row[f"R@{k}"] = round(float(np.mean(results[m_name][f"R@{k}"])), 4)
        row[f"NDCG@{k}"] = round(float(np.mean(results[m_name][f"NDCG@{k}"])), 4)
    summary_table.append(row)

df_res = pd.DataFrame(summary_table)
print("\n" + "=" * 85)
print("RECOMMENDATION BENCHMARK WITH COURSE-ENROLLED CANDIDATE FILTERING")
print("=" * 85)
print(df_res.to_string(index=False))
print("=" * 85 + "\n")
