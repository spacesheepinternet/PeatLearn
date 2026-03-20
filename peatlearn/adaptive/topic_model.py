#!/usr/bin/env python3
"""
Corpus-driven topic modeling and query classification for PeatLearn.

Goals:
- Build topics from the Ray Peat corpus using TF-IDF + clustering
- Label topics with top keywords and representative documents
- Provide fast, no-API assignment of a user query to a topic via centroid similarity
- Hybrid option: if RAG returns source files, infer topic by majority from doc-to-cluster mapping

Artifacts written to disk:
- data/models/topics/topics.json: cluster metadata and doc->cluster mapping
- data/models/topics/tfidf_vectorizer.joblib: fitted scikit-learn TfidfVectorizer
- data/models/topics/cluster_centroids.npy: centroid matrix (clusters x vocab)
- data/reports/topic_clusters_report.md: human-readable summary report

Dependencies: scikit-learn, numpy, scipy, joblib
"""
from __future__ import annotations

import os
import json
import glob
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import TruncatedSVD
    from joblib import dump, load
except Exception as e:
    # Lazy import error message to avoid crashing callers that only use type hints
    TfidfVectorizer = None  # type: ignore
    KMeans = None  # type: ignore
    silhouette_score = None  # type: ignore
    TruncatedSVD = None  # type: ignore
    dump = None  # type: ignore
    load = None  # type: ignore


@dataclass
class TopicCluster:
    cluster_id: int
    label: str
    top_keywords: List[str]
    doc_ids: List[str]


class CorpusTopicModel:
    def __init__(
        self,
        corpus_glob: str = "data/processed/**/**/*.txt",
        model_dir: str = "data/models/topics",
        max_features: int = 50000,
        min_df: int = 2,
        max_df: float = 0.6,
        n_components_svd: Optional[int] = 300,
    ) -> None:
        self.corpus_glob = corpus_glob
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.vectorizer_path = self.model_dir / "tfidf_vectorizer.joblib"
        self.centroids_path = self.model_dir / "cluster_centroids.npy"
        self.topics_json_path = self.model_dir / "topics.json"
        self.whitelist_path = self.model_dir / "cluster_whitelist.json"
        self.report_path = Path("data/reports/topic_clusters_report.md")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.n_components_svd = n_components_svd

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.kmeans: Optional[KMeans] = None
        self.svd: Optional[TruncatedSVD] = None
        self.doc_ids: List[str] = []
        self.doc_to_cluster: Dict[str, int] = {}
        self.clusters: List[TopicCluster] = []
        self.centroids: Optional[np.ndarray] = None
        self.cluster_whitelist: Optional[set[int]] = None

    def _load_corpus(self) -> Tuple[List[str], List[str]]:
        paths = sorted(glob.glob(self.corpus_glob, recursive=True))
        docs: List[str] = []
        doc_ids: List[str] = []
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                    if txt and len(txt.split()) > 30:
                        docs.append(txt)
                        doc_ids.append(os.path.relpath(p))
            except Exception:
                continue
        return docs, doc_ids

    def build(self, k: Optional[int] = None, k_min: int = 12, k_max: int = 36) -> None:
        assert TfidfVectorizer is not None, "scikit-learn is required. Install scikit-learn and joblib."
        docs, doc_ids = self._load_corpus()
        if not docs:
            raise RuntimeError("No documents found for topic modeling. Check corpus_glob path.")
        self.doc_ids = doc_ids

        # Vectorize
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=(1, 2),
        )
        X = self.vectorizer.fit_transform(docs)

        # Optional dimensionality reduction for stability
        if self.n_components_svd and X.shape[1] > self.n_components_svd:
            self.svd = TruncatedSVD(n_components=self.n_components_svd, random_state=42)
            X_reduced = self.svd.fit_transform(X)
        else:
            X_reduced = X

        # Choose k via silhouette if not provided
        if k is None:
            best_k = None
            best_score = -1
            candidates = list(range(k_min, min(k_max, max(13, len(docs)//50)) + 1))
            for kk in candidates:
                km = KMeans(n_clusters=kk, random_state=42, n_init=10)
                labels = km.fit_predict(X_reduced)
                try:
                    score = silhouette_score(X_reduced, labels)
                except Exception:
                    score = -1
                if score > best_score:
                    best_score, best_k = score, kk
            k = best_k or max(12, min(24, len(docs)//75))

        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X_reduced)

        # Compute centroids in TF-IDF (or reduced) space
        if isinstance(X_reduced, np.ndarray):
            centroids = np.vstack([
                X_reduced[labels == c].mean(axis=0) if np.any(labels == c) else np.zeros((X_reduced.shape[1],), dtype=float)
                for c in range(k)
            ])
        else:
            # sparse matrix
            centroids = np.vstack([
                X_reduced[labels == c].mean(axis=0) if np.any(labels == c) else np.zeros((X_reduced.shape[1],), dtype=float)
                for c in range(k)
            ])
        self.centroids = centroids

        # Top keywords per cluster
        feature_names = (self.svd.components_ if self.svd else self.vectorizer.get_feature_names_out())  # type: ignore
        # If SVD used, we need to approximate keyword importances by mapping cluster centroid back to tfidf space
        if self.svd is not None:
            # centroid in reduced space -> pseudo-term weights via inverse transform
            pseudo_terms = self.svd.inverse_transform(centroids)
            vocab_terms = self.vectorizer.get_feature_names_out()
            def top_terms(row: np.ndarray, n=12):
                idx = np.argsort(row)[::-1][:n]
                return [vocab_terms[i] for i in idx]
            cluster_keywords = [top_terms(pseudo_terms[c]) for c in range(k)]
        else:
            # no svd: use tfidf features directly
            def top_terms_dense(center: np.ndarray, n=12):
                idx = np.argsort(center)[::-1][:n]
                return [feature_names[i] for i in idx]
            cluster_keywords = [top_terms_dense(centroids[c]) for c in range(k)]

        # Build clusters metadata
        clusters: List[TopicCluster] = []
        doc_to_cluster: Dict[str, int] = {}
        for c in range(k):
            doc_ids_c = [doc_ids[i] for i, lb in enumerate(labels) if lb == c]
            label = ", ".join(cluster_keywords[c][:3]) if cluster_keywords[c] else f"Cluster {c}"
            clusters.append(TopicCluster(cluster_id=c, label=label, top_keywords=cluster_keywords[c], doc_ids=doc_ids_c))
            for di in doc_ids_c:
                doc_to_cluster[di] = c

        self.clusters = clusters
        self.doc_to_cluster = doc_to_cluster

        # Persist artifacts
        if dump is not None:
            dump(self.vectorizer, self.vectorizer_path)
            if self.svd is not None:
                dump(self.svd, self.model_dir / "svd.joblib")
        np.save(self.centroids_path, centroids)
        with open(self.topics_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "clusters": [c.__dict__ for c in clusters],
                "doc_to_cluster": doc_to_cluster,
            }, f, indent=2)

        # Report
        sizes = [len(c.doc_ids) for c in clusters]
        with open(self.report_path, "w", encoding="utf-8") as rf:
            rf.write("# Topic Clusters Report\n\n")
            rf.write(f"Documents: {len(docs)}\n")
            rf.write(f"Clusters (k): {k}\n")
            rf.write(f"Avg documents/cluster: {np.mean(sizes):.1f} (median {np.median(sizes):.1f})\n\n")
            for c in clusters:
                rf.write(f"## Cluster {c.cluster_id}: {c.label}\n")
                rf.write(f"Top keywords: {', '.join(c.top_keywords[:12])}\n\n")
                rf.write(f"Examples ({min(5, len(c.doc_ids))}):\n")
                for ex in c.doc_ids[:5]:
                    rf.write(f"- {ex}\n")
                rf.write("\n")

    def load(self) -> None:
        assert Path(self.topics_json_path).exists(), "topics.json not found. Build the topic model first."
        self.vectorizer = load(self.vectorizer_path) if load is not None else None
        svd_path = self.model_dir / "svd.joblib"
        self.svd = load(svd_path) if svd_path.exists() and load is not None else None
        self.centroids = np.load(self.centroids_path)
        with open(self.topics_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.doc_to_cluster = data.get("doc_to_cluster", {})
        self.clusters = [TopicCluster(**c) for c in data.get("clusters", [])]
        # Load whitelist if present
        if self.whitelist_path.exists():
            try:
                wl = json.load(open(self.whitelist_path, 'r', encoding='utf-8'))
                self.cluster_whitelist = set(wl.get('whitelist_ids', []))
            except Exception:
                self.cluster_whitelist = None

    def assign_topic_from_rag_sources(self, source_files: List[str]) -> Optional[TopicCluster]:
        if not source_files or not self.doc_to_cluster or not self.clusters:
            return None
        votes: Dict[int, int] = {}
        for sf in source_files:
            # we stored relpaths; try to match suffix
            matched = None
            for doc, c in self.doc_to_cluster.items():
                if doc.endswith(sf) or sf.endswith(doc):
                    matched = c
                    break
            if matched is not None:
                votes[matched] = votes.get(matched, 0) + 1
        if not votes:
            return None
        # Pick the most voted cluster, honoring whitelist if present
        ranked = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        for cid, _ in ranked:
            if self.cluster_whitelist is None or cid in self.cluster_whitelist:
                return next((c for c in self.clusters if c.cluster_id == cid), None)
        return None

    def assign_topic_from_text(self, text: str) -> Optional[TopicCluster]:
        if self.vectorizer is None or self.centroids is None:
            raise RuntimeError("Topic model not loaded. Call load() first.")
        x = self.vectorizer.transform([text])
        if self.svd is not None:
            x = self.svd.transform(x)
        # Compute cosine similarity to centroids
        x_dense = x if isinstance(x, np.ndarray) else x.toarray()
        cent = self.centroids
        # normalize
        x_norm = x_dense / (np.linalg.norm(x_dense, axis=1, keepdims=True) + 1e-9)
        cent_norm = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
        sims = x_norm @ cent_norm.T
        order = np.argsort(sims[0])[::-1]
        for idx in order:
            cid = int(idx)
            if self.cluster_whitelist is None or cid in self.cluster_whitelist:
                return next((c for c in self.clusters if c.cluster_id == cid), None)
        return None

    def jargon_score(self, text: str, cluster: TopicCluster, top_n: int = 15) -> float:
        """Measure how much of the cluster's top vocabulary appears in the text (proxy for nomenclature depth).

        Uses exact substring, then falls back to fuzzy matching to tolerate minor misspellings.
        """
        text_lower = text.lower()
        vocab = [t.lower() for t in cluster.top_keywords[:top_n]]
        hits = 0
        try:
            from difflib import SequenceMatcher
        except Exception:
            SequenceMatcher = None  # type: ignore
        for term in vocab:
            if term in text_lower:
                hits += 1
            elif SequenceMatcher is not None:
                # approximate match: high similarity and similar length
                ratio = SequenceMatcher(None, term, text_lower).find_longest_match(0, len(term), 0, len(text_lower))
                # Compute normalized similarity by matching segment
                if ratio.size > 0:
                    seg = text_lower[ratio.b: ratio.b + ratio.size]
                    # similarity of term vs matched segment
                    sim = SequenceMatcher(None, term, seg).ratio()
                    if sim >= 0.85:
                        hits += 1
        return hits / max(1, len(vocab))

    def similarity_to_cluster(self, text: str, cluster: TopicCluster) -> float:
        """Compute cosine similarity between query TF-IDF (optionally SVD) and a cluster centroid."""
        if self.vectorizer is None or self.centroids is None:
            return 0.0
        try:
            x = self.vectorizer.transform([text])
            if self.svd is not None:
                x = self.svd.transform(x)
            x_dense = x if isinstance(x, np.ndarray) else x.toarray()
            v = x_dense[0]
            c = self.centroids[cluster.cluster_id]
            num = float(np.dot(v, c))
            den = (np.linalg.norm(v) * np.linalg.norm(c) + 1e-9)
            return max(0.0, min(1.0, num / den))
        except Exception:
            return 0.0
