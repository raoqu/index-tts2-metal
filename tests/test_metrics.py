from __future__ import annotations

import numpy as np

from metal_indextts2.metrics import compare_arrays


def test_compare_arrays_reports_similarity():
    report = compare_arrays(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert report["shape_match"] is True
    assert report["max_abs_error"] == 1.0
    assert report["cosine_similarity"] < 1.0
