from __future__ import annotations

import numpy as np

from classic_sar_metrics import edge_preservation_index, m_index_paper


def test_epi_identity_is_one() -> None:
    noisy = np.arange(64, dtype=np.float64).reshape(8, 8) / 63.0
    assert edge_preservation_index(noisy, noisy) == 1.0


def test_epi_uses_diagonal_neighbours() -> None:
    noisy = np.array([[0.0, 1.0], [2.0, 4.0]])
    prediction = np.array([[0.0, 2.0], [1.0, 2.0]])
    assert edge_preservation_index(noisy, prediction) == 0.5


def test_m_identity_is_explicitly_undefined_without_fallback() -> None:
    rng = np.random.default_rng(7)
    noisy = rng.gamma(shape=1.0, scale=0.2, size=(64, 64))
    result = m_index_paper(noisy, noisy, shuffles=4, seed=42)
    assert not result.valid
    assert np.isnan(result.value)
    assert result.selected_windows == 0
    assert result.reason == "no_textureless_window_satisfies_paper_tolerance"


def test_m_is_finite_and_reproducible_for_matching_speckle_ratio() -> None:
    rng = np.random.default_rng(11)
    clean = np.full((96, 96), 0.3, dtype=np.float64)
    noisy = clean * rng.gamma(shape=1.0, scale=1.0, size=clean.shape)
    first = m_index_paper(noisy, clean, shuffles=4, seed=19)
    second = m_index_paper(noisy, clean, shuffles=4, seed=19)
    assert first.valid
    assert first.selected_windows > 0
    assert np.isfinite(first.value)
    assert first.value == second.value
