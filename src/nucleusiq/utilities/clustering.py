# src/nucleusiq/utilities/clustering.py

from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def cluster_questions(questions: List[str], num_clusters: int = 5) -> Dict[int, List[str]]:
    """
    Clusters questions into the specified number of clusters using KMeans.

    Args:
        questions (List[str]): List of questions to cluster.
        num_clusters (int): Number of clusters.

    Returns:
        Dict[int, List[str]]: A dictionary mapping cluster IDs to lists of questions.
    """
    if not questions:
        return {}

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(questions)

    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(X)
    labels = km.labels_

    clusters: Dict[int, List[str]] = {}
    for label, question in zip(labels, questions):
        clusters.setdefault(label, []).append(question)

    return clusters
