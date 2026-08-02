"""Potencia del plan de análisis, calculada en vez de estimada.

La nota de potencia del preregistro cita cifras concretas, así que tienen que
ser reproducibles: este módulo las produce. No hay fórmula cerrada para un
Wilcoxon de rangos con signo aplicado a coeficientes de correlación, de modo
que la etapa 2 se resuelve por simulación Monte Carlo (simular bajo la verdad
asumida, analizar con el test exacto que se va a usar, repetir), mientras que
Spearman y el binomial de H5 sí tienen forma cerrada o exacta.

Dos cosas que este módulo existe para no volver a olvidar:

- **La potencia se calcula al alfa que el criterio realmente exige.** Los
  criterios de H1/H2 deciden a q < 0,05 bajo BH sobre familias de 8, no a
  p < 0,05. Calcular a 0,05 y decidir a q es el error clásico, y es el que
  hacía inadvertidamente la primera versión de la nota.
- **El Wilcoxon tiene un suelo discreto anterior a la potencia.** Con n celdas
  el p bilateral más pequeño que puede producir es 2**(1-n). Si ese mínimo
  supera el alfa corregido, el test no puede rechazar tenga el efecto que
  tenga, y ninguna curva de potencia lo muestra.

Ejecutar con ``uv run python src/power_analysis.py`` para reimprimir las tablas
que cita el plan.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# Familias confirmatorias de 8 tests (una por métrica de gradiente).
FAMILY_SIZE = 8
C_BY = sum(1.0 / i for i in range(1, FAMILY_SIZE + 1))  # c(8) ≈ 2,718

ALPHA_RAW = 0.05
ALPHA_BH = ALPHA_RAW / FAMILY_SIZE  # peor caso de BH = Bonferroni
ALPHA_BY = ALPHA_RAW / (FAMILY_SIZE * C_BY)

N_SIM = 20_000
SEED = 20260801


def min_attainable_p(n: int) -> float:
    """p bilateral mínimo que el Wilcoxon de rangos con signo puede producir.

    Se alcanza cuando los n valores comparten signo: una sola de las 2**n
    asignaciones de signo iguala o supera el estadístico, y el bilateral la
    duplica.
    """
    return 2.0 ** (1 - n)


def can_reject(n: int, alpha: float) -> bool:
    """¿Puede el test rechazar a este alfa, con el efecto más extremo posible?"""
    return min_attainable_p(n) < alpha


def _wilson_ci(hits: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """IC de Wilson al 95% para la proporción de Monte Carlo."""
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def power_stage2(
    n_cells: int,
    median_rho: float,
    sd_rho: float = 0.25,
    alpha: float = ALPHA_BH,
    n_sim: int = N_SIM,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Potencia del Wilcoxon de etapa 2 (rho de celda contra 0, bilateral).

    Los rho se simulan normales recortados a (-1, 1), que es su soporte. Es una
    aproximación optimista: la distribución real será asimétrica y más dispersa
    en las celdas difíciles.

    Devuelve (potencia, ic_bajo, ic_alto) con el IC de Monte Carlo.
    """
    rng = np.random.default_rng(seed)
    draws = np.clip(rng.normal(median_rho, sd_rho, (n_sim, n_cells)), -0.999, 0.999)
    hits = sum(
        stats.wilcoxon(row, alternative="two-sided", method="auto").pvalue < alpha
        for row in draws
    )
    lo, hi = _wilson_ci(hits, n_sim)
    return hits / n_sim, lo, hi


def mde_stage2(
    n_cells: int,
    alpha: float = ALPHA_BH,
    target: float = 0.80,
    sd_rho: float = 0.25,
    n_sim: int = 4_000,
    seed: int = SEED,
) -> float:
    """Mediana de rho detectable a `target` de potencia (bisección)."""
    lo, hi = 0.0, 0.8
    for _ in range(16):
        mid = (lo + hi) / 2
        if power_stage2(n_cells, mid, sd_rho, alpha, n_sim, seed)[0] < target:
            lo = mid
        else:
            hi = mid
    return hi


def power_h4(
    n_cells: int,
    true_median_d: float,
    sd_d: float,
    delta: float = 0.10,
    alpha: float = ALPHA_RAW,
    n_sim: int = N_SIM,
    seed: int = SEED,
) -> float:
    """Potencia del contraste de no-inferioridad de H4.

    d_i = |rho_i@0,50| - |rho_i@0,10| por celda; se rechaza H0: mediana(d) >= delta
    a favor de mediana(d) < delta. `true_median_d = 0` es la saturación real,
    que es lo que H4 quiere poder afirmar.
    """
    rng = np.random.default_rng(seed)
    draws = rng.normal(true_median_d, sd_d, (n_sim, n_cells))
    hits = sum(
        stats.wilcoxon(row - delta, alternative="less", method="auto").pvalue < alpha
        for row in draws
    )
    return hits / n_sim


def power_spearman(n: int, rho: float, alpha: float = ALPHA_RAW) -> float:
    """Potencia del test de Spearman (Fisher-z con la inflación 1,06 de Spearman).

    `n` es el número de runs de la celda para el valor nominal, o el n efectivo
    (entre 8 y 40 aquí, por el clustering de LR) para la lectura honesta.
    """
    se = 1.06 / np.sqrt(n - 3)
    z = np.arctanh(rho) / se
    crit = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(crit - z) + stats.norm.cdf(-crit - z))


def power_binomial_concordance(
    n_pairs: int, true_concordance: float, alpha: float = ALPHA_RAW
) -> float:
    """Potencia del binomial exacto bilateral de H5 contra 0,5.

    Exacta, no simulada: se suma la masa binomial sobre los recuentos que
    rechazan.
    """
    counts = np.arange(n_pairs + 1)
    rejects = np.array(
        [stats.binomtest(int(k), n_pairs, 0.5).pvalue < alpha for k in counts]
    )
    return float(stats.binom.pmf(counts, n_pairs, true_concordance)[rejects].sum())


def _signed_rank_null_counts(n: int) -> np.ndarray:
    """Recuentos de la nula exacta de W+ para n observaciones.

    counts[w] = número de subconjuntos de {1..n} que suman w, de 2**n en total.
    Producto polinómico iterativo: O(n * n**2 / 2).
    """
    counts = np.zeros(n * (n + 1) // 2 + 1, dtype=np.float64)
    counts[0] = 1.0
    for rank in range(1, n + 1):
        counts[rank:] += counts[:-rank].copy()
    return counts


def hodges_lehmann(x, confidence: float = 0.95) -> tuple[float, float, float]:
    """Pseudomediana de Hodges-Lehmann con su IC exacto.

    Es el estimador que el Wilcoxon de rangos con signo localiza de hecho; la
    mediana muestral es otro estimador y solo coincide bajo simetría. El plan
    reporta ambos.

    Devuelve (estimador, ic_bajo, ic_alto).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    walsh = np.sort(
        np.array([(x[i] + x[j]) / 2 for i in range(n) for j in range(i, n)])
    )
    estimate = float(np.median(walsh))

    counts = _signed_rank_null_counts(n)
    tail = np.cumsum(counts) / 2.0**n  # P(W+ <= w)
    alpha = 1 - confidence
    # k = mayor recuento con masa acumulada <= alpha/2; el IC recorta k medias
    # de Walsh por cada lado.
    k = int(np.searchsorted(tail, alpha / 2, side="right"))
    k = min(k, walsh.size // 2)
    return estimate, float(walsh[k]), float(walsh[-1 - k])


def _main() -> None:
    """Reimprime las tablas que cita §Nota de potencia del plan."""
    print(f"alfa crudo {ALPHA_RAW}  |  BH8 {ALPHA_BH:.5f}  |  BY8 {ALPHA_BY:.5f}"
          f"  (c(8) = {C_BY:.3f})\n")

    print("Etapa 2: potencia a mediana de rho = 0,30, sd = 0,25")
    print(f"{'celdas':>7} {'crudo':>8} {'BH8':>8} {'BY8':>8}   suelo discreto")
    for n in (24, 20, 18, 16, 14, 12, 10, 9, 8):
        row = [power_stage2(n, 0.30, alpha=a)[0] for a in (ALPHA_RAW, ALPHA_BH, ALPHA_BY)]
        floor = "BH y BY ok" if can_reject(n, ALPHA_BY) else (
            "BY IMPOSIBLE" if can_reject(n, ALPHA_BH) else "BH y BY IMPOSIBLES")
        print(f"{n:>7} {row[0]:>8.3f} {row[1]:>8.3f} {row[2]:>8.3f}   {floor}")

    print("\nH2: potencia con 24 celdas bajo BH, según la mediana de la PARCIAL")
    for m in (0.30, 0.25, 0.20, 0.15, 0.10):
        print(f"  parcial {m:.2f} -> {power_stage2(24, m)[0]:.3f}")

    print(f"\nMDE (24 celdas, 80% potencia): crudo {mde_stage2(24, ALPHA_RAW):.3f}"
          f"  BH8 {mde_stage2(24, ALPHA_BH):.3f}  BY8 {mde_stage2(24, ALPHA_BY):.3f}")

    print("\nH4: potencia bajo saturación real (mediana d = 0), 24 celdas")
    for sd in (0.10, 0.15, 0.20, 0.30):
        print(f"  sd(d) {sd:.2f} -> {power_h4(24, 0.0, sd):.3f}")

    print("\nH5: binomial exacto sobre 12 pares")
    for conc in (0.75, 0.85, 0.95, 1.00):
        print(f"  concordancia {conc:.2f} -> {power_binomial_concordance(12, conc):.3f}")

    print("\nEtapa 1: Spearman por celda")
    for rho in (0.30, 0.44):
        print(f"  rho {rho:.2f}: n=40 -> {power_spearman(40, rho):.3f}"
              f"   n efectivo 8 -> {power_spearman(8, rho):.3f}")


if __name__ == "__main__":
    _main()
