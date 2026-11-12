import numpy as np
import pytest
from scipy import stats

from bayes_canais import diagnosticos as diag
from bayes_canais import hierarquico as hier

# As oito escolas (Rubin, 1981; Gelman et al., BDA, seção 5.5): o exemplo de livro do modelo
# normal-normal com variância conhecida. Efeito estimado e erro padrão de cada escola.
ESCOLAS_EFEITO = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
ESCOLAS_ERRO = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


def _quantis_da_grade(grade: np.ndarray, densidade: np.ndarray, probabilidades: list[float]) -> np.ndarray:
    acumulada = np.concatenate([[0.0], np.cumsum((densidade[1:] + densidade[:-1]) / 2 * np.diff(grade))])
    return np.interp(probabilidades, acumulada / acumulada[-1], grade)


# --------------------------------------------------------------------------------------- normal truncada


def test_normal_truncada_respeita_os_limites_e_bate_com_a_media_teorica():
    gerador = np.random.default_rng(0)
    n = 200_000
    so_piso = hier.normal_truncada(gerador, np.full(n, 1.0), np.full(n, 2.0), np.full(n, 2.5))
    assert so_piso.min() >= 2.5
    assert so_piso.mean() == pytest.approx(stats.truncnorm.mean(0.75, np.inf, loc=1.0, scale=2.0), abs=0.01)

    com_teto = hier.normal_truncada(gerador, np.full(n, 1.0), np.full(n, 2.0), np.full(n, 2.5), np.full(n, 3.5))
    assert com_teto.min() >= 2.5 and com_teto.max() <= 3.5
    assert com_teto.mean() == pytest.approx(stats.truncnorm.mean(0.75, 1.25, loc=1.0, scale=2.0), abs=0.005)
    assert com_teto.std() == pytest.approx(stats.truncnorm.std(0.75, 1.25, loc=1.0, scale=2.0), rel=0.02)


def test_normal_truncada_nao_quebra_com_piso_muito_acima_da_media():
    gerador = np.random.default_rng(1)
    amostra = hier.normal_truncada(gerador, np.zeros(1000), np.ones(1000), np.full(1000, 12.0))
    assert np.isfinite(amostra).all()
    assert amostra.min() >= 12.0 and amostra.max() < 13.0  # a cauda além de 12 desvios cai muito rápido


# --------------------------------------------------------------------------------------- resposta analítica


def test_com_variancias_conhecidas_o_amostrador_bate_com_a_formula_do_modelo_normal_normal():
    gerador = np.random.default_rng(2)
    n = np.array([3, 5, 8, 12, 20])
    sigma, tau, media_mu, desvio_mu = 1.5, 0.8, 2.0, 3.0
    grupo = np.repeat(np.arange(5), n)
    y = gerador.normal(np.array([1.0, 2.0, 2.5, 3.0, 4.0])[grupo], sigma)
    media = np.array([y[grupo == j].mean() for j in range(5)])
    variancia = sigma**2 / n

    posteriori = hier.amostrar(
        hier.estrutura(y, grupo), tau_fixo=tau, sigma_fixo=sigma, media_mu=media_mu, desvio_mu=desvio_mu, cadeias=4, iteracoes=6000, aquecimento=200, semente=3
    )

    # mu: fórmula fechada com theta integrado fora.
    mu_media, mu_desvio = hier.posteriori_de_mu(media, variancia, tau, media_mu, desvio_mu)
    assert posteriori.juntas("mu")[:, 0].mean() == pytest.approx(mu_media, abs=0.02)
    assert posteriori.juntas("mu")[:, 0].std() == pytest.approx(mu_desvio, rel=0.04)

    # theta: dado mu é a fórmula conjugada; como ela é linear em mu, a incerteza de mu entra somada.
    theta_media, theta_desvio = hier.posteriori_conjugada(media, variancia, mu_media, tau)
    peso_de_mu = (1 / tau**2) / (1 / variancia + 1 / tau**2)
    theta_desvio_total = np.sqrt(theta_desvio**2 + (peso_de_mu * mu_desvio) ** 2)
    assert posteriori.juntas("theta").mean(axis=0) == pytest.approx(theta_media, abs=0.02)
    assert posteriori.juntas("theta").std(axis=0) == pytest.approx(theta_desvio_total, rel=0.04)


def test_posteriori_conjugada_encolhe_mais_o_grupo_com_menos_dados():
    media, _ = hier.posteriori_conjugada(np.array([10.0, 10.0]), np.array([4.0, 0.25]), mu=0.0, tau=1.0)
    assert media[0] == pytest.approx(2.0)  # variância 4 contra priori 1: fica com 1/5 da própria média
    assert media[1] == pytest.approx(8.0)  # variância 0,25: fica com 4/5
    assert media[0] < media[1] < 10.0


@pytest.mark.parametrize("priori", [hier.uniforme(100.0), hier.semi_cauchy(5.0)], ids=["uniforme", "semi-Cauchy"])
def test_passo_de_metropolis_reproduz_a_marginal_de_tau_nas_oito_escolas(priori):
    # Com sigma conhecido, p(tau | y) tem forma fechada a menos de uma integral em uma dimensão.
    # A grade faz essa integral; o amostrador tem de chegar aos mesmos quantis.
    escolas = hier.estrutura(ESCOLAS_EFEITO, np.arange(8), escala=np.arange(8))
    posteriori = hier.amostrar(escolas, priori, sigma_fixo=ESCOLAS_ERRO, media_mu=0.0, desvio_mu=1000.0, cadeias=4, iteracoes=15000, aquecimento=1000, semente=4)

    grade = np.linspace(1e-4, 100.0, 20001)
    densidade = hier.densidade_marginal_de_tau(grade, priori, ESCOLAS_EFEITO, ESCOLAS_ERRO**2, 0.0, 1000.0)
    probabilidades = [0.25, 0.5, 0.75, 0.9]
    esperado = _quantis_da_grade(grade, densidade, probabilidades)
    obtido = np.quantile(posteriori.juntas("tau"), probabilidades)
    assert obtido == pytest.approx(esperado, rel=0.08, abs=0.25)

    # Média a posteriori da escola A, integrando a fórmula conjugada sobre a marginal de tau.
    medias_dado_tau = []
    for tau in grade[::20]:
        mu_media, _ = hier.posteriori_de_mu(ESCOLAS_EFEITO, ESCOLAS_ERRO**2, tau, 0.0, 1000.0)
        medias_dado_tau.append(hier.posteriori_conjugada(ESCOLAS_EFEITO, ESCOLAS_ERRO**2, mu_media, tau)[0][0])
    esperada_a = np.trapezoid(np.array(medias_dado_tau) * densidade[::20], grade[::20]) / np.trapezoid(densidade[::20], grade[::20])
    assert posteriori.juntas("theta")[:, 0].mean() == pytest.approx(esperada_a, abs=0.35)

    assert 0.2 < posteriori.aceitacao_tau.mean() < 0.7
    assert diag.rhat(posteriori.tau) < 1.02
    assert diag.rhat(posteriori.theta[:, :, 0]) < 1.01


def test_oito_escolas_com_priori_uniforme_chega_aos_valores_do_livro():
    # BDA (3ª ed., tabela 5.3 e seção 5.5): a média geral fica perto de 8 e a escola A, perto de 11.
    escolas = hier.estrutura(ESCOLAS_EFEITO, np.arange(8), escala=np.arange(8))
    posteriori = hier.amostrar(escolas, hier.uniforme(200.0), sigma_fixo=ESCOLAS_ERRO, media_mu=0.0, desvio_mu=1000.0, cadeias=4, iteracoes=15000, aquecimento=1000, semente=5)
    assert posteriori.juntas("mu")[:, 0].mean() == pytest.approx(8.0, abs=0.6)
    assert posteriori.juntas("theta")[:, 0].mean() == pytest.approx(11.0, abs=1.0)
    assert np.median(posteriori.juntas("theta")[:, 2]) == pytest.approx(7.0, abs=1.0)  # escola C


# --------------------------------------------------------------------------------------- dados simulados


def _simular(gerador, grupos_por_bloco=12, n=25, mu=(6.0, 3.0), tau=0.5, sigma=(0.5, 2.0)):
    bloco = np.repeat([0, 1], grupos_por_bloco)
    theta = gerador.normal(np.array(mu)[bloco], tau)
    grupo = np.repeat(np.arange(len(bloco)), n)
    y = gerador.normal(theta[grupo], np.array(sigma)[bloco][grupo])
    return hier.estrutura(y, grupo, bloco=bloco, escala=bloco), theta


def test_recupera_os_parametros_de_dados_simulados():
    dados, theta = _simular(np.random.default_rng(6))
    posteriori = hier.amostrar(dados, hier.semi_normal(1.0), cadeias=4, iteracoes=2500, aquecimento=500, semente=7)

    assert np.median(posteriori.juntas("sigma"), axis=0) == pytest.approx([0.5, 2.0], rel=0.08)
    assert 0.3 < np.median(posteriori.juntas("tau")) < 0.8  # verdadeiro: 0,5 (24 grupos: ainda incerto)
    inferior, superior = np.quantile(posteriori.juntas("mu"), [0.005, 0.995], axis=0)
    assert (inferior < [6.0, 3.0]).all() and (superior > [6.0, 3.0]).all()

    inferior, superior = np.quantile(posteriori.juntas("theta"), [0.025, 0.975], axis=0)
    cobertura = ((inferior < theta) & (theta < superior)).mean()
    assert cobertura >= 0.85  # intervalos de 95% em 24 grupos

    for j in (0, 23):
        assert diag.rhat(posteriori.theta[:, :, j]) < 1.01
    assert diag.rhat(posteriori.tau) < 1.03


def test_modelo_hierarquico_erra_menos_que_a_media_do_grupo_quando_ha_poucos_dados():
    # O motivo de existir do agrupamento parcial: com 3 posts por grupo e grupos parecidos,
    # a média encolhida fica mais perto da verdade que a média de cada grupo sozinho.
    gerador = np.random.default_rng(8)
    dados, theta = _simular(gerador, grupos_por_bloco=15, n=3, mu=(5.0, 5.0), tau=0.3, sigma=(1.5, 1.5))
    media_do_grupo = np.array([dados.y[dados.grupo == j].mean() for j in range(dados.grupos)])
    posteriori = hier.amostrar(dados, hier.semi_normal(1.0), cadeias=4, iteracoes=2000, aquecimento=500, semente=9)
    erro_hierarquico = np.abs(posteriori.juntas("theta").mean(axis=0) - theta).mean()
    erro_sozinho = np.abs(media_do_grupo - theta).mean()
    assert erro_hierarquico < 0.7 * erro_sozinho


def test_tau_enorme_devolve_a_media_de_cada_grupo_e_grupo_unico_devolve_a_media_geral():
    gerador = np.random.default_rng(10)
    grupo = np.repeat(np.arange(4), [6, 9, 12, 15])
    y = gerador.normal(np.array([1.0, 3.0, 5.0, 7.0])[grupo], 1.0)
    medias = np.array([y[grupo == j].mean() for j in range(4)])

    sozinho = hier.amostrar(hier.estrutura(y, grupo, bloco=np.arange(4)), tau_fixo=hier.TAU_SEM_AGRUPAMENTO, cadeias=4, iteracoes=4000, aquecimento=300, semente=11)
    assert sozinho.juntas("theta").mean(axis=0) == pytest.approx(medias, abs=0.03)
    assert np.isnan(sozinho.aceitacao_tau).all()  # tau fixo: não há passo de Metropolis

    junto = hier.amostrar(hier.estrutura(y, np.zeros(len(y), dtype=int)), tau_fixo=hier.TAU_SEM_AGRUPAMENTO, cadeias=4, iteracoes=4000, aquecimento=300, semente=12)
    assert junto.juntas("theta")[:, 0].mean() == pytest.approx(y.mean(), abs=0.03)


# --------------------------------------------------------------------------------------- censura


def test_censura_a_direita_recupera_a_media_que_o_valor_observado_subestima():
    # Tudo que passaria de 5,5 é registrado como "pelo menos 5,5" (censura de tipo I).
    gerador = np.random.default_rng(13)
    y = gerador.normal(5.0, 1.0, 600)
    censurado = y > 5.5
    observado = np.where(censurado, 5.5, y)
    assert observado.mean() < 4.9  # tratar o piso como valor final puxa a média para baixo

    posteriori = hier.amostrar(
        hier.estrutura(observado, np.zeros(600, dtype=int), censurado=censurado), tau_fixo=hier.TAU_SEM_AGRUPAMENTO, cadeias=4, iteracoes=3000, aquecimento=500, semente=14
    )
    assert posteriori.juntas("theta")[:, 0].mean() == pytest.approx(5.0, abs=0.12)
    assert np.median(posteriori.juntas("sigma")[:, 0]) == pytest.approx(1.0, abs=0.10)
    assert diag.rhat(posteriori.theta[:, :, 0]) < 1.01


def test_censura_em_intervalo_recupera_media_e_desvio_de_dados_vistos_so_por_faixa():
    # De cada valor só se sabe em que faixa de largura 1 ele caiu.
    gerador = np.random.default_rng(15)
    y = gerador.normal(3.3, 1.2, 800)
    piso = np.floor(y)
    posteriori = hier.amostrar(
        hier.estrutura(piso, np.zeros(800, dtype=int), censurado=np.ones(800, dtype=bool), teto=piso + 1.0),
        tau_fixo=hier.TAU_SEM_AGRUPAMENTO,
        cadeias=4,
        iteracoes=3000,
        aquecimento=500,
        semente=16,
    )
    assert posteriori.juntas("theta")[:, 0].mean() == pytest.approx(3.3, abs=0.12)
    assert np.median(posteriori.juntas("sigma")[:, 0]) == pytest.approx(1.2, abs=0.10)


def test_estrutura_recusa_entradas_incoerentes():
    with pytest.raises(ValueError):
        hier.estrutura([1.0, 2.0], [0, 1, 1])
    with pytest.raises(ValueError):
        hier.estrutura([1.0, 2.0], [0, 1], censurado=[True, False], teto=[0.5, np.inf])  # teto abaixo do piso
    with pytest.raises(ValueError):
        hier.Estrutura(np.zeros(2), np.zeros(2, dtype=bool), np.array([0, 1]), np.array([0]), np.array([0]), np.full(2, np.inf))


# --------------------------------------------------------------------------------------- previsão e reprodução


def test_log_densidade_preditiva_bate_com_a_normal_analitica():
    gerador = np.random.default_rng(17)
    y = gerador.normal(4.0, 1.0, 30)
    sigma, tau, mu = 1.0, 2.0, 3.0
    posteriori = hier.amostrar(
        hier.estrutura(y, np.zeros(30, dtype=int)), tau_fixo=tau, sigma_fixo=sigma, media_mu=mu, desvio_mu=1e-6, cadeias=4, iteracoes=5000, aquecimento=200, semente=18
    )
    media, desvio = hier.posteriori_conjugada(np.array([y.mean()]), np.array([sigma**2 / 30]), mu, tau)
    for novo in (2.0, 4.0, 6.5):
        esperado = stats.norm.logpdf(novo, media[0], np.sqrt(desvio[0] ** 2 + sigma**2))
        assert hier.log_densidade_preditiva(posteriori, novo, 0) == pytest.approx(esperado, abs=0.02)


def test_replicas_tem_o_desenho_dos_dados_e_a_dispersao_do_modelo():
    gerador = np.random.default_rng(19)
    grupo = np.repeat([0, 1], 40)
    y = gerador.normal(np.array([2.0, 6.0])[grupo], 0.7)
    posteriori = hier.amostrar(hier.estrutura(y, grupo, bloco=[0, 1]), tau_fixo=hier.TAU_SEM_AGRUPAMENTO, cadeias=4, iteracoes=1500, aquecimento=300, semente=20)
    replicas = hier.replicar(posteriori, np.random.default_rng(21), 500)
    assert replicas.shape == (500, 80)
    assert replicas[:, grupo == 0].mean() == pytest.approx(y[grupo == 0].mean(), abs=0.1)
    assert replicas[:, grupo == 1].std(axis=1).mean() == pytest.approx(0.7, abs=0.1)


def test_a_mesma_semente_reproduz_as_mesmas_amostras():
    dados, _ = _simular(np.random.default_rng(22), grupos_por_bloco=3, n=5)
    a = hier.amostrar(dados, cadeias=2, iteracoes=200, aquecimento=100, semente=23)
    b = hier.amostrar(dados, cadeias=2, iteracoes=200, aquecimento=100, semente=23)
    c = hier.amostrar(dados, cadeias=2, iteracoes=200, aquecimento=100, semente=24)
    assert np.array_equal(a.theta, b.theta) and np.array_equal(a.tau, b.tau)
    assert not np.array_equal(a.theta, c.theta)
