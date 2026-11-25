"""
Bipartite Student-Resource Graph Learning and Recommendation.
Constructs a heterogeneous bipartite graph using NetworkX.
Implements Graph Convolution / LightGCN-style message passing representation.
Evaluates with Precision@K, Recall@K, and NDCG@K.
"""

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


class StudentResourceBipartiteGraph:
    def __init__(
        self, raw_data_dir: Path = Path("data/raw"), artifacts_dir: Path = Path("artifacts")
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.graph = nx.Graph()

    def build_bipartite_graph(self, min_interactions: int = 2):
        df_vle = pd.read_csv(self.raw_data_dir / "vle.csv")
        df_student_vle = pd.read_csv(self.raw_data_dir / "studentVle.csv")
        df_student_info = pd.read_csv(self.raw_data_dir / "studentInfo.csv")

        # 1. Add Resource Nodes
        for _, row in df_vle.iterrows():
            site_id = f"res_{int(row['id_site'])}"
            self.graph.add_node(
                site_id,
                bipartite=1,
                node_type="resource",
                module=row["code_module"],
                activity_type=str(row["activity_type"]),
            )

        # 2. Add Student Nodes
        for _, row in df_student_info.iterrows():
            stud_id = f"stud_{int(row['id_student'])}"
            self.graph.add_node(
                stud_id,
                bipartite=0,
                node_type="student",
                education=str(row["highest_education"]),
                outcome=str(row["final_result"]),
            )

        # 3. Aggregate Student-Resource Interactions
        interaction_agg = (
            df_student_vle.groupby(["id_student", "id_site"])["sum_click"].sum().reset_index()
        )
        interaction_agg = interaction_agg[interaction_agg["sum_click"] >= min_interactions]

        for _, row in interaction_agg.iterrows():
            stud_node = f"stud_{int(row['id_student'])}"
            res_node = f"res_{int(row['id_site'])}"
            if self.graph.has_node(stud_node) and self.graph.has_node(res_node):
                weight = float(np.log1p(row["sum_click"]))
                self.graph.add_edge(
                    stud_node, res_node, weight=weight, clicks=int(row["sum_click"])
                )

        print(
            f"Graph constructed: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges."
        )
        return self.graph

    def train_test_split_interactions(self, test_ratio: float = 0.20, seed: int = 42):
        np.random.seed(seed)
        student_nodes = [
            n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "student"
        ]
        train_edges = []
        test_edges = []

        train_graph = nx.Graph()
        # copy nodes
        for n, d in self.graph.nodes(data=True):
            train_graph.add_node(n, **d)

        for s in student_nodes:
            neighbors = list(self.graph.neighbors(s))
            if len(neighbors) < 2:
                for res in neighbors:
                    e_data = self.graph.get_edge_data(s, res)
                    train_graph.add_edge(s, res, **e_data)
                    train_edges.append((s, res))
                continue

            np.random.shuffle(neighbors)
            n_test = max(1, int(len(neighbors) * test_ratio))
            test_res = neighbors[:n_test]
            train_res = neighbors[n_test:]

            for res in train_res:
                e_data = self.graph.get_edge_data(s, res)
                train_graph.add_edge(s, res, **e_data)
                train_edges.append((s, res))

            for res in test_res:
                test_edges.append((s, res))

        print(f"Train/Test split: {len(train_edges)} train edges, {len(test_edges)} test edges.")
        return train_graph, test_edges

    def compute_lightgcn_style_embeddings(
        self, train_graph: nx.Graph, dim: int = 32, num_layers: int = 2
    ):
        """
        Computes LightGCN-style multi-layer neighborhood aggregation representation.
        E^(k+1) = D^(-1/2) * A * D^(-1/2) * E^(k)
        Final embedding = (1 / (K+1)) * sum_k E^(k)
        """
        nodes = list(train_graph.nodes())
        node2idx = {n: i for i, n in enumerate(nodes)}
        N = len(nodes)

        # Initial random embedding layer E^(0)
        np.random.seed(42)
        E0 = np.random.normal(0, 0.1, size=(N, dim))

        # Compute normalized adjacency matrix
        A = nx.adjacency_matrix(train_graph, nodelist=nodes, weight="weight").astype(np.float32)
        degrees = np.array(A.sum(axis=1)).flatten()
        deg_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
        deg_inv_sqrt[degrees == 0] = 0.0

        # Symmetrically normalized adjacency: D^(-1/2) A D^(-1/2)
        from scipy.sparse import diags

        D_mat = diags(deg_inv_sqrt)
        A_norm = D_mat.dot(A).dot(D_mat)

        all_layers = [E0]
        cur_E = E0
        for _ in range(num_layers):
            cur_E = A_norm.dot(cur_E)
            all_layers.append(cur_E)

        final_embeddings = np.mean(all_layers, axis=0)
        embeddings_dict = {nodes[i]: final_embeddings[i] for i in range(N)}
        return embeddings_dict, node2idx

    def evaluate_ranking(self, embeddings_dict: dict, test_edges: list, k_list: list = [5, 10, 20]):
        # Group test edges by student
        test_by_student = {}
        for s, r in test_edges:
            test_by_student.setdefault(s, set()).add(r)

        resource_nodes = [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "resource" and n in embeddings_dict
        ]
        resource_emb = np.array([embeddings_dict[r] for r in resource_nodes])

        precisions = {k: [] for k in k_list}
        recalls = {k: [] for k in k_list}
        ndcgs = {k: [] for k in k_list}

        # Evaluate over a sample of 250 active students for efficiency
        sampled_students = list(test_by_student.keys())[:250]
        for s in sampled_students:
            if s not in embeddings_dict:
                continue
            true_items = test_by_student[s]
            s_emb = embeddings_dict[s]

            # Dot product ranking
            scores = np.dot(resource_emb, s_emb)
            top_indices = np.argsort(-scores)

            for k in k_list:
                top_k_items = [resource_nodes[idx] for idx in top_indices[:k]]
                hits = [1 if item in true_items else 0 for item in top_k_items]
                num_hits = sum(hits)

                # Precision@K
                prec = num_hits / k
                precisions[k].append(prec)

                # Recall@K
                rec = num_hits / len(true_items)
                recalls[k].append(rec)

                # NDCG@K
                dcg = sum(h / math.log2(idx + 2) for idx, h in enumerate(hits))
                idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(true_items))))
                ndcg = (dcg / idcg) if idcg > 0 else 0.0
                ndcgs[k].append(ndcg)

        metrics = {}
        for k in k_list:
            metrics[f"precision@{k}"] = round(float(np.mean(precisions[k])), 4)
            metrics[f"recall@{k}"] = round(float(np.mean(recalls[k])), 4)
            metrics[f"ndcg@{k}"] = round(float(np.mean(ndcgs[k])), 4)

        return metrics

    def run_pipeline(self):
        self.build_bipartite_graph()
        train_graph, test_edges = self.train_test_split_interactions()
        embeddings, _ = self.compute_lightgcn_style_embeddings(train_graph)
        metrics = self.evaluate_ranking(embeddings, test_edges)

        # Course-filtered candidate evaluation
        course_filtered_metrics = self.evaluate_course_filtered(embeddings, test_edges)

        # Save graph metadata & evaluation
        output_metrics = {
            "graph_stats": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "student_nodes": len(
                    [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "student"]
                ),
                "resource_nodes": len(
                    [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "resource"]
                ),
            },
            "unconstrained_ranking_metrics": metrics,
            "course_filtered_benchmark": course_filtered_metrics,
        }

        # Persist embeddings for fast inference in backend
        np.savez_compressed(
            self.artifacts_dir / "models" / "graph_embeddings.npz",
            **{k: v for k, v in embeddings.items()},
        )

        with open(self.artifacts_dir / "metrics" / "graph_metrics.json", "w") as f:
            json.dump(output_metrics, f, indent=2)

        print("Graph pipeline finished successfully. Metrics:")
        print(json.dumps(output_metrics, indent=2))
        return output_metrics

    def evaluate_course_filtered(
        self, embeddings_dict: dict, test_edges: list, k_list: list = [5, 10, 20]
    ):
        df_vle = pd.read_csv("data/raw/vle.csv")
        df_student_info = pd.read_csv("data/raw/studentInfo.csv")
        df_student_vle = pd.read_csv("data/raw/studentVle.csv")

        module_resources = df_vle.groupby("code_module")["id_site"].unique().to_dict()
        pop_counts = df_student_vle.groupby("id_site")["id_student"].nunique().to_dict()

        test_by_student = {}
        for s, r in test_edges:
            test_by_student.setdefault(s, set()).add(r)

        results = {
            "lightgcn": {f"p@{k}": [] for k in k_list},
            "popularity": {f"p@{k}": [] for k in k_list},
            "random": {f"p@{k}": [] for k in k_list},
        }
        for m in list(results):
            for k in k_list:
                results[m][f"r@{k}"] = []
                results[m][f"ndcg@{k}"] = []

        sampled_students = list(test_by_student.keys())[:250]
        for s_node in sampled_students:
            if s_node not in embeddings_dict:
                continue
            s_id = int(s_node.replace("stud_", ""))
            true_items = {int(r.replace("res_", "")) for r in test_by_student[s_node]}
            stud_modules = df_student_info[df_student_info["id_student"] == s_id][
                "code_module"
            ].unique()
            if len(stud_modules) == 0:
                continue

            candidates = []
            for mod in stud_modules:
                candidates.extend(module_resources.get(mod, []))
            candidates = list(set(candidates))
            if len(candidates) < 10:
                continue

            true_in_course = true_items.intersection(set(candidates))
            if len(true_in_course) == 0:
                continue

            s_emb = embeddings_dict[s_node]

            # LightGCN ranking
            def get_score(site_id, s_vec=s_emb):
                r_emb = embeddings_dict.get(f"res_{site_id}")
                return float(np.dot(s_vec, r_emb)) if r_emb is not None else -999.0

            lg_ranked = sorted(candidates, key=get_score, reverse=True)
            pop_ranked = sorted(candidates, key=lambda x: pop_counts.get(x, 0), reverse=True)
            np.random.seed(s_id)
            rand_ranked = list(np.random.permutation(candidates))

            rankings = {
                "lightgcn": lg_ranked,
                "popularity": pop_ranked,
                "random": rand_ranked,
            }

            for m_name, ranked in rankings.items():
                for k in k_list:
                    top_k = ranked[:k]
                    hits = [1 if item in true_in_course else 0 for item in top_k]
                    n_hits = sum(hits)
                    p = n_hits / k
                    r = n_hits / len(true_in_course)
                    dcg = sum(h / math.log2(idx + 2) for idx, h in enumerate(hits))
                    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(true_in_course))))
                    ndcg = (dcg / idcg) if idcg > 0 else 0.0

                    results[m_name][f"p@{k}"].append(p)
                    results[m_name][f"r@{k}"].append(r)
                    results[m_name][f"ndcg@{k}"].append(ndcg)

        benchmark_summary = {}
        for m_name in results:
            benchmark_summary[m_name] = {}
            for k in k_list:
                benchmark_summary[m_name][f"precision@{k}"] = round(
                    float(np.mean(results[m_name][f"p@{k}"])), 4
                )
                benchmark_summary[m_name][f"recall@{k}"] = round(
                    float(np.mean(results[m_name][f"r@{k}"])), 4
                )
                benchmark_summary[m_name][f"ndcg@{k}"] = round(
                    float(np.mean(results[m_name][f"ndcg@{k}"])), 4
                )

        return benchmark_summary


if __name__ == "__main__":
    bipartite = StudentResourceBipartiteGraph()
    bipartite.run_pipeline()
