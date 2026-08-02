"""Comprobaciones analíticas del módulo de potencia.

Como el resto de la suite: respuestas conocidas a mano donde las hay, y
calibración bajo la nula donde el resultado es simulado. La prueba que de
verdad valida el arnés de Monte Carlo es la de tamaño: bajo H0 la tasa de
rechazo tiene que ser el alfa nominal, no otra cosa.
"""

import numpy as np
import pytest
from scipy import stats

from power_analysis import (
    ALPHA_BH,
    ALPHA_BY,
    ALPHA_RAW,
    C_BY,
    _signed_rank_null_counts,
    can_reject,
    hodges_lehmann,
    min_attainable_p,
    power_binomial_concordance,
    power_spearman,
    power_stage2,
)


def test_by_constant_matches_harmonic_sum():
    assert C_BY == pytest.approx(1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5 + 1 / 6 + 1 / 7 + 1 / 8)
    assert ALPHA_BH == pytest.approx(0.00625)
    assert ALPHA_BY == pytest.approx(0.05 / (8 * C_BY))


def test_min_attainable_p_known_values():
    # Con n valores del mismo signo, una sola de las 2**n asignaciones iguala
    # el estadístico; el bilateral la duplica.
    assert min_attainable_p(6) == pytest.approx(0.03125)
    assert min_attainable_p(8) == pytest.approx(0.0078125)
    assert min_attainable_p(12) == pytest.approx(0.00048828125)


def test_min_attainable_p_agrees_with_scipy():
    for n in (6, 8, 10, 12):
        allsame = np.arange(1.0, n + 1)  # todos positivos: el caso extremo
        got = stats.wilcoxon(allsame, alternative="two-sided", method="exact").pvalue
        assert got == pytest.approx(min_attainable_p(n))


def test_discrete_floor_boundaries():
    """El suelo que hace inútil al test antes de que la potencia importe."""
    assert not can_reject(8, ALPHA_BH)   # 0,0078 > 0,00625
    assert can_reject(9, ALPHA_BH)       # 0,0039 < 0,00625
    assert not can_reject(9, ALPHA_BY)   # 0,0039 > 0,00230
    assert can_reject(10, ALPHA_BY)      # 0,00195 < 0,00230
    assert can_reject(6, ALPHA_RAW)      # 0,031 < 0,05


def test_signed_rank_null_counts_sums_to_2n():
    for n in (1, 2, 5, 12):
        assert _signed_rank_null_counts(n).sum() == pytest.approx(2.0**n)


def test_signed_rank_null_counts_small_case_by_hand():
    # Subconjuntos de {1,2}: sumas 0, 1, 2, 3 -> un subconjunto cada una.
    assert list(_signed_rank_null_counts(2)) == [1.0, 1.0, 1.0, 1.0]


def test_hodges_lehmann_point_estimate_by_hand():
    # Medias de Walsh de [1,2,3]: 1, 1.5, 2, 2, 2.5, 3 -> mediana 2.
    est, lo, hi = hodges_lehmann([1.0, 2.0, 3.0])
    assert est == pytest.approx(2.0)
    assert lo <= est <= hi


def test_hodges_lehmann_equals_median_under_symmetry():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    est, _, _ = hodges_lehmann(x)
    assert est == pytest.approx(np.median(x))


def test_hodges_lehmann_ci_covers_at_nominal_rate():
    rng = np.random.default_rng(7)
    truth = 0.30
    covered = sum(
        lo <= truth <= hi
        for lo, hi in (
            hodges_lehmann(rng.normal(truth, 0.25, 24))[1:] for _ in range(400)
        )
    )
    assert covered / 400 >= 0.90


def test_stage2_has_correct_size_under_the_null():
    """Bajo H0 la tasa de rechazo debe ser el alfa nominal (validación del arnés)."""
    power, _, _ = power_stage2(24, median_rho=0.0, alpha=ALPHA_RAW, n_sim=4000, seed=1)
    assert power == pytest.approx(ALPHA_RAW, abs=0.02)


def test_stage2_power_increases_with_cells_and_effect():
    p12 = power_stage2(12, 0.30, n_sim=2000)[0]
    p24 = power_stage2(24, 0.30, n_sim=2000)[0]
    assert p12 < p24
    assert power_stage2(24, 0.15, n_sim=2000)[0] < p24


def test_stage2_reproduces_the_thresholds_the_plan_cites():
    """18 celdas entran; 12 no. Es la regla de matriz incompleta de §Censura."""
    assert power_stage2(18, 0.30, n_sim=4000)[0] > 0.90
    assert power_stage2(12, 0.30, n_sim=4000)[0] < 0.80


def test_binomial_concordance_is_one_at_perfect_agreement():
    assert power_binomial_concordance(12, 1.00) == pytest.approx(1.0)


def test_binomial_concordance_is_zero_when_alpha_is_unreachable():
    # Con 5 pares el p bilateral mínimo es 2*(1/32) = 0,0625 > 0,05.
    assert power_binomial_concordance(5, 1.00) == pytest.approx(0.0)


def test_binomial_concordance_at_chance_equals_size():
    # Bajo concordancia 0,5 la potencia es el tamaño del test, y por ser
    # discreto queda por debajo del alfa nominal.
    assert power_binomial_concordance(12, 0.50) <= ALPHA_RAW


def test_spearman_power_monotone_and_calibrated():
    assert power_spearman(40, 0.0) == pytest.approx(ALPHA_RAW, abs=1e-9)
    assert power_spearman(40, 0.30) < power_spearman(40, 0.44) < power_spearman(40, 0.50)
    assert power_spearman(8, 0.30) < power_spearman(40, 0.30)
