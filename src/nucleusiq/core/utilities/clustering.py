# src/nucleusiq/utilities/clustering.py

from typing import List, Dict


def cluster_questions(questions: List[str], num_clusters: int = 5) -> Dict[int, List[str]]:
    """
    Clusters questions into the specified number of clusters using KMeans.

    Args:
        questions (List[str]): List of questions to cluster.
        num_clusters (int): Number of clusters.

    Returns:
        Dict[int, List[str]]: A dictionary mapping cluster IDs to lists of questions.

    Requires:
        scikit-learn: pip install scikit-learn
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError(
            "Auto Chain-of-Thought clustering requires scikit-learn. "
            "Install with: pip install scikit-learn"
        )

    if not questions:
        return {}

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(questions)

    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(X)
    labels = km.labels_

    # Type check: labels_ is always set after fit(), but type checker doesn't know this
    if labels is None:
        raise RuntimeError("KMeans labels not available after fitting")

    clusters: Dict[int, List[str]] = {}
    for label, question in zip(labels, questions):
        clusters.setdefault(label, []).append(question)

    return clusters
