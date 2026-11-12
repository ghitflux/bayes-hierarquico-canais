import numpy as np
import pytest
from scipy import stats

from bayes_canais import bandit


def _mundos(theta: list[float], sigma: float, mundos: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.tile(np.array(theta, dtype=float), (mundos, 1))
    return theta, np.full_like(theta, sigma)


def test_posteriori_normal_gama_inversa_bate_com_a_conta_feita_a_mao():
    # Três observações (1, 2, 6) com priori m0=0, k0=1, a0=2, b0=3.
    m, k, a, b = bandit.posteriori_normal_gama_inversa(np.array([3.0]), np.array([9.0]), np.array([41.0]), m0=0.0, k0=1.0, a0=2.0, b0=3.0)
    assert k[0] == pytest.approx(4.0)
    assert m[0] == pytest.approx(9.0 / 4.0)
    assert a[0] == pytest.approx(3.5)
    # b = b0 + metade da soma dos desvios (14) + k0 * n * (média - m0)² / (2 * k) = 3 + 7 + 27/8
    assert b[0] == pytest.approx(3.0 + 7.0 + 27.0 / 8.0)


def test_posteriori_sem_dados_e_a_propria_priori():
    m, k, a, b = bandit.posteriori_normal_gama_inversa(np.zeros(2), np.zeros(2), np.zeros(2), m0=4.0, k0=0.1, a0=1.0, b0=1.0)
    assert m == pytest.approx([4.0, 4.0]) and k == pytest.approx([0.1, 0.1])
    assert a == pytest.approx([1.0, 1.0]) and b == pytest.approx([1.0, 1.0])


def test_arrependimento_da_reparticao_igual_e_a_conta_fechada():
    theta, sigma = _mundos([np.log1p(100.0), np.log1p(60.0), np.log1p(10.0)], 1.0, 5)
    escolhas = bandit.simular(bandit.ReparticaoIgual(), theta, sigma, 30, semente=0)
    acumulado = bandit.arrependimento(escolhas, theta)
    assert (np.bincount(escolhas[0], minlength=3) == 10).all()
    assert acumulado[:, -1] == pytest.approx(10 * (0 + 40 + 90))  # dez posts em cada canal
    assert (np.diff(acumulado, axis=1) >= 0).all()  # arrependimento acumulado nunca diminui


def test_quem_sempre_escolhe_o_melhor_tem_arrependimento_zero():
    theta, _ = _mundos([1.0, 3.0, 2.0], 1.0, 4)
    assert bandit.arrependimento(np.full((4, 20), 1), theta)[:, -1] == pytest.approx(0.0)


@pytest.mark.parametrize("posts_por_canal", [1, 4])
def test_chance_de_travar_no_canal_errado_bate_com_a_formula(posts_por_canal):
    # Dois canais com diferença d e desvio sigma: depois de n posts em cada, a média do pior passa
    # a do melhor com probabilidade Phi(-d / (sigma * raiz(2 / n))).
    d, sigma = 0.8, 1.5
    theta, sigmas = _mundos([d, 0.0], sigma, 40_000)
    escolhas = bandit.simular(bandit.ApostarNoLider(posts_por_canal), theta, sigmas, 2 * posts_por_canal + 6, semente=1)
    travou_no_errado = (bandit.canal_final(escolhas, 2) == 1).mean()
    esperado = stats.norm.cdf(-d / (sigma * np.sqrt(2 / posts_por_canal)))
    assert travou_no_errado == pytest.approx(esperado, abs=0.01)
    # Depois de decidir, não muda mais de canal.
    depois = escolhas[:, 2 * posts_por_canal :]
    assert (depois == depois[:, :1]).all()


def test_thompson_encontra_o_melhor_canal_e_perde_menos_que_a_reparticao_igual():
    theta, sigma = _mundos([5.0, 4.0, 3.0], 0.7, 400)
    thompson = bandit.simular(bandit.Thompson(), theta, sigma, 200, semente=2)
    igual = bandit.simular(bandit.ReparticaoIgual(), theta, sigma, 200, semente=3)
    assert (bandit.canal_final(thompson, 3) == 0).mean() > 0.98
    assert (thompson == 0).mean() > 0.85  # a maior parte dos posts vai para o melhor canal
    perda_thompson = bandit.arrependimento(thompson, theta)[:, -1].mean()
    perda_igual = bandit.arrependimento(igual, theta)[:, -1].mean()
    assert perda_thompson < 0.25 * perda_igual
    # A exploração diminui com o tempo: menos posts fora do melhor na segunda metade.
    assert (thompson[:, 100:] != 0).mean() < (thompson[:, :100] != 0).mean()


def test_thompson_testa_todos_os_canais_antes_de_se_concentrar():
    theta, sigma = _mundos([5.0, 4.9, 4.8, 4.7], 1.0, 200)
    escolhas = bandit.simular(bandit.Thompson(), theta, sigma, 60, semente=4)
    canais_testados = np.array([len(np.unique(linha)) for linha in escolhas])
    assert (canais_testados == 4).mean() > 0.95


def test_simulacao_e_reprodutivel_pela_semente():
    theta, sigma = _mundos([2.0, 2.5], 1.0, 50)
    a = bandit.simular(bandit.Thompson(), theta, sigma, 40, semente=5)
    b = bandit.simular(bandit.Thompson(), theta, sigma, 40, semente=5)
    assert np.array_equal(a, b)
