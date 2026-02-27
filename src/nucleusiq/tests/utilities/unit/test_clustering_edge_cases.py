"""Edge-case coverage tests for utilities/clustering.py."""

from __future__ import annotations

import builtins
import types

import pytest
from nucleusiq.utilities.clustering import cluster_questions


def test_cluster_questions_import_error(monkeypatch):
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("no sklearn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(ImportError, match="requires scikit-learn"):
        cluster_questions(["a", "b"])


def test_cluster_questions_labels_none_raises_runtime(monkeypatch):
    class _Vec:
        def fit_transform(self, _questions):
            return [[1], [2]]

    class _KM:
        def __init__(self, n_clusters, random_state):
            self.n_clusters = n_clusters
            self.random_state = random_state
            self.labels_ = None

        def fit(self, _x):
            return None

    fake_cluster = types.SimpleNamespace(KMeans=_KM)
    fake_text = types.SimpleNamespace(TfidfVectorizer=lambda stop_words: _Vec())

    monkeypatch.setitem(__import__("sys").modules, "sklearn.cluster", fake_cluster)
    monkeypatch.setitem(
        __import__("sys").modules,
        "sklearn.feature_extraction.text",
        fake_text,
    )

    with pytest.raises(RuntimeError, match="labels not available"):
        cluster_questions(["q1", "q2"], num_clusters=2)

