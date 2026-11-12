"""Diagnósticos de convergência de cadeias de Markov: R-hat e tamanho efetivo de amostra.

As duas funções recebem uma matriz `(cadeias, iterações)` com as amostras de um parâmetro, já sem
o aquecimento. As fórmulas são as de Gelman et al., *Bayesian Data Analysis* (3ª ed., seção 11.4 e 11.5):
cada cadeia é partida ao meio, para que uma cadeia que ainda está andando também seja flagrada.
"""

from __future__ import annotations

import numpy as np


def _partir_ao_meio(cadeias: np.ndarray) -> np.ndarray:
    """Cada cadeia vira duas (primeira e segunda metade): (C, S) -> (2C, S // 2)."""
    cadeias = np.asarray(cadeias, dtype=float)
    if cadeias.ndim != 2:
        raise ValueError("esperado um array (cadeias, iterações)")
    metade = cadeias.shape[1] // 2
    if metade < 2:
        raise ValueError("cadeias curtas demais para o diagnóstico")
    return np.concatenate([cadeias[:, :metade], cadeias[:, -metade:]], axis=0)


def _variancias(partes: np.ndarray) -> tuple[float, float]:
    """Variância dentro das cadeias (W) e estimativa conservadora da variância a posteriori (var+)."""
    n = partes.shape[1]
    dentro = float(partes.var(axis=1, ddof=1).mean())
    entre = float(n * partes.mean(axis=1).var(ddof=1))
    return dentro, (n - 1) / n * dentro + entre / n


def rhat(cadeias: np.ndarray) -> float:
    """Fator de redução de escala (R-hat com cadeias partidas).

    Compara a variância entre as cadeias com a variância dentro delas. Perto de 1, as cadeias
    contam a mesma história; acima de 1,01 (regra atual) ou 1,1 (regra antiga), ainda não misturaram.
    """
    dentro, total = _variancias(_partir_ao_meio(cadeias))
    if dentro == 0:
        return float("nan")
    return float(np.sqrt(total / dentro))


def _autocovariancia(serie: np.ndarray) -> np.ndarray:
    """Autocovariância (viesada, divisor n) em todas as defasagens, por FFT."""
    n = len(serie)
    centrada = serie - serie.mean()
    tamanho = 2 ** int(np.ceil(np.log2(2 * n)))
    transformada = np.fft.rfft(centrada, tamanho)
    return np.fft.irfft(transformada * np.conj(transformada), tamanho)[:n] / n


def tamanho_efetivo(cadeias: np.ndarray) -> float:
    """Tamanho efetivo de amostra: a quantas amostras independentes a cadeia autocorrelacionada equivale.

    Soma as autocorrelações combinadas das cadeias, aos pares de defasagens, e para no primeiro par
    negativo (sequência positiva inicial de Geyer), forçando os pares a não crescerem.
    """
    partes = _partir_ao_meio(cadeias)
    m, n = partes.shape
    dentro, total = _variancias(partes)
    if total == 0:
        return float("nan")
    autocov = np.array([_autocovariancia(parte) for parte in partes]) * n / (n - 1)
    rho = 1.0 - (dentro - autocov.mean(axis=0)) / total
    rho[0] = 1.0
    soma, anterior = 0.0, np.inf
    for t in range(0, n - 1, 2):
        par = rho[t] + rho[t + 1]
        if par < 0:
            break
        par = min(par, anterior)
        soma += par
        anterior = par
    tau = max(-1.0 + 2.0 * soma, 1.0 / np.log10(m * n))
    return float(m * n / tau)


def erro_de_monte_carlo(cadeias: np.ndarray) -> float:
    """Erro padrão da média a posteriori devido ao número finito de amostras."""
    return float(np.std(cadeias, ddof=1) / np.sqrt(tamanho_efetivo(cadeias)))
