import numpy as np
import pytest

from bayes_canais import diagnosticos as diag


def _ar1(gerador: np.random.Generator, cadeias: int, n: int, rho: float) -> np.ndarray:
    """Cadeias AR(1) estacionárias com variância 1: a autocorrelação na defasagem k é rho**k."""
    x = np.empty((cadeias, n))
    x[:, 0] = gerador.normal(size=cadeias)
    ruido = gerador.normal(size=(cadeias, n)) * np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + ruido[:, t]
    return x


def test_amostras_independentes_tem_rhat_um_e_tamanho_efetivo_igual_ao_numero_de_amostras():
    cadeias = np.random.default_rng(0).normal(size=(4, 5000))
    assert diag.rhat(cadeias) == pytest.approx(1.0, abs=0.005)
    assert diag.tamanho_efetivo(cadeias) == pytest.approx(20000, rel=0.10)


@pytest.mark.parametrize("rho", [0.5, 0.9])
def test_tamanho_efetivo_de_cadeia_ar1_bate_com_a_formula(rho):
    # Para AR(1), o tamanho efetivo teórico é N * (1 - rho) / (1 + rho).
    cadeias = _ar1(np.random.default_rng(1), 4, 20000, rho)
    esperado = cadeias.size * (1 - rho) / (1 + rho)
    assert diag.tamanho_efetivo(cadeias) == pytest.approx(esperado, rel=0.15)
    assert diag.rhat(cadeias) < 1.01


def test_rhat_acusa_cadeias_presas_em_lugares_diferentes():
    gerador = np.random.default_rng(2)
    cadeias = gerador.normal(size=(4, 2000)) + np.array([[0.0], [0.0], [3.0], [3.0]])
    assert diag.rhat(cadeias) > 1.5
    assert diag.tamanho_efetivo(cadeias) < 50  # quatro cadeias que não concordam valem quase nada


def test_rhat_com_cadeias_partidas_acusa_cadeia_que_ainda_esta_andando():
    # Todas as cadeias sobem juntas: sem partir ao meio, elas pareceriam concordar.
    gerador = np.random.default_rng(3)
    cadeias = gerador.normal(size=(4, 2000)) + np.linspace(0, 4, 2000)
    assert diag.rhat(cadeias) > 1.3


def test_erro_de_monte_carlo_de_amostras_independentes_e_desvio_sobre_raiz_de_n():
    cadeias = np.random.default_rng(4).normal(0, 2, size=(4, 5000))
    assert diag.erro_de_monte_carlo(cadeias) == pytest.approx(2 / np.sqrt(20000), rel=0.10)


def test_diagnosticos_recusam_entrada_curta_ou_no_formato_errado():
    with pytest.raises(ValueError):
        diag.rhat(np.zeros(10))
    with pytest.raises(ValueError):
        diag.tamanho_efetivo(np.zeros((4, 3)))
