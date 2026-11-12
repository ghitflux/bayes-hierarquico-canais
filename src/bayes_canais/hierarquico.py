"""Modelo normal hierárquico em escala log, com amostrador próprio (Gibbs e Metropolis dentro de Gibbs).

O modelo, para o post i do canal j:

    y_i      ~ Normal(theta_j, sigma_k²)        y = log(1 + views); k é o grupo de dispersão do canal
    theta_j  ~ Normal(mu_b, tau²)               b é o bloco do canal (a plataforma); tau é o desvio entre canais
    mu_b     ~ Normal(media_mu, desvio_mu²)     priori larga
    sigma_k² ~ Gama-Inversa(a_sigma, b_sigma)   priori fraca
    tau      ~ priori escolhida (semi-normal, semi-Cauchy, uniforme): é a que mais pesa com poucos canais

Post jovem (views ainda subindo) entra como observação censurada: sabe-se só que o valor final fica
entre o observado (piso) e um teto, que pode ser infinito. O amostrador sorteia esse valor final a
cada passo (aumento de dados).

Cada volta do amostrador:
  1. valores finais dos posts censurados  | theta, sigma        normal truncada          (Gibbs)
  2. mu     | tau, sigma, y (theta integrado fora)               normal                   (Gibbs)
  3. tau    | mu, sigma, y (theta integrado fora)                passeio em log(tau)      (Metropolis)
  4. theta  | mu, tau, sigma, y                                  normal                   (Gibbs)
  5. sigma² | theta, y                                           gama-inversa             (Gibbs)

Os passos 2 e 3 usam a verossimilhança com theta integrado (média do canal ~ Normal(mu, tau² + sigma²/n)).
Isso evita o "funil" do Gibbs ingênuo, em que tau pequeno prende os theta e os theta juntos prendem tau.

Com `tau_fixo` muito grande, o mesmo código estima cada grupo sozinho (sem agrupamento); com um grupo
só, estima a média comum (agrupamento total). Assim as três formas de estimar usam a mesma verossimilhança.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import special, stats

TAU_SEM_AGRUPAMENTO = 1e3  # desvio entre grupos tão grande que a priori dos theta fica plana


# --------------------------------------------------------------------------------------- prioris para tau


@dataclass(frozen=True)
class Priori:
    """Priori para tau (desvio entre canais, em escala log). `log_densidade` pode omitir a constante."""

    nome: str
    log_densidade: Callable[[np.ndarray], np.ndarray]
    amostrar: Callable[[np.random.Generator, int], np.ndarray]


def semi_normal(escala: float = 1.0) -> Priori:
    """Normal(0, escala²) restrita a tau > 0: cauda leve, puxa para diferenças pequenas entre canais."""
    return Priori(
        f"semi-normal({escala:g})",
        lambda tau: -0.5 * (np.asarray(tau) / escala) ** 2,
        lambda gerador, n: np.abs(gerador.normal(0.0, escala, n)),
    )


def semi_cauchy(escala: float = 1.0) -> Priori:
    """Cauchy(0, escala) restrita a tau > 0: cauda pesada, deixa passar diferenças grandes."""
    return Priori(
        f"semi-Cauchy({escala:g})",
        lambda tau: -np.log1p((np.asarray(tau) / escala) ** 2),
        lambda gerador, n: np.abs(escala * gerador.standard_cauchy(n)),
    )


def uniforme(limite: float = 5.0) -> Priori:
    """Uniforme(0, limite): não prefere nenhum valor de tau dentro da faixa."""
    return Priori(
        f"uniforme(0, {limite:g})",
        lambda tau: np.where(np.asarray(tau) <= limite, 0.0, -np.inf),
        lambda gerador, n: gerador.uniform(0.0, limite, n),
    )


# --------------------------------------------------------------------------------------- dados e resultado


@dataclass(frozen=True)
class Estrutura:
    """Os dados e quem pertence a quê.

    y          (N,) valor em log; para observação censurada, o piso
    censurado  (N,) True quando só se sabe que o valor final está entre y e teto
    grupo      (N,) índice do grupo (canal) de cada observação, de 0 a J-1
    bloco      (J,) índice da média comum (plataforma) de cada grupo
    escala     (J,) índice do desvio sigma de cada grupo
    teto       (N,) limite superior do valor final das observações censuradas (infinito = sem teto)
    """

    y: np.ndarray
    censurado: np.ndarray
    grupo: np.ndarray
    bloco: np.ndarray
    escala: np.ndarray
    teto: np.ndarray

    def __post_init__(self) -> None:
        if not (len(self.y) == len(self.censurado) == len(self.grupo) == len(self.teto)):
            raise ValueError("y, censurado, grupo e teto precisam ter o mesmo tamanho")
        if len(self.bloco) != len(self.escala) or int(np.max(self.grupo)) >= len(self.bloco):
            raise ValueError("bloco e escala precisam ter uma entrada por grupo")
        if np.any(self.teto[self.censurado] <= self.y[self.censurado]):
            raise ValueError("o teto de uma observação censurada precisa ficar acima do piso")

    @property
    def grupos(self) -> int:
        return len(self.bloco)

    @property
    def blocos(self) -> int:
        return int(np.max(self.bloco)) + 1

    @property
    def escalas(self) -> int:
        return int(np.max(self.escala)) + 1


def estrutura(y, grupo, bloco=None, escala=None, censurado=None, teto=None) -> Estrutura:
    """Monta a `Estrutura` com padrões simples: um bloco só, um sigma só, nada censurado."""
    y = np.asarray(y, dtype=float)
    grupo = np.asarray(grupo, dtype=int)
    grupos = int(grupo.max()) + 1
    return Estrutura(
        y=y,
        censurado=np.zeros(len(y), dtype=bool) if censurado is None else np.asarray(censurado, dtype=bool),
        grupo=grupo,
        bloco=np.zeros(grupos, dtype=int) if bloco is None else np.asarray(bloco, dtype=int),
        escala=np.zeros(grupos, dtype=int) if escala is None else np.asarray(escala, dtype=int),
        teto=np.full(len(y), np.inf) if teto is None else np.asarray(teto, dtype=float),
    )


@dataclass(frozen=True)
class Posteriori:
    """Amostras a posteriori, sem o aquecimento. Formato `(cadeias, iterações, dimensão)`; tau é `(cadeias, iterações)`."""

    theta: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    tau: np.ndarray
    aceitacao_tau: np.ndarray  # taxa de aceitação do passo de Metropolis, por cadeia (NaN se tau é fixo)
    estrutura: Estrutura

    def juntas(self, nome: str) -> np.ndarray:
        """As cadeias empilhadas: `(cadeias * iterações, dimensão)`."""
        valores = getattr(self, nome)
        return valores.reshape(-1, *valores.shape[2:])

    def sigma_do_grupo(self) -> np.ndarray:
        """Sigma de cada grupo, amostra a amostra: `(cadeias * iterações, grupos)`."""
        return self.juntas("sigma")[:, self.estrutura.escala]


# --------------------------------------------------------------------------------------- peças do amostrador


def normal_truncada(gerador: np.random.Generator, media: np.ndarray, desvio: np.ndarray, piso: np.ndarray, teto: np.ndarray | float = np.inf) -> np.ndarray:
    """Sorteia de Normal(media, desvio²) condicionada a ficar entre piso e teto, pela inversa da função de distribuição.

    Trabalha com o logaritmo das probabilidades de cauda, para não perder precisão quando o piso
    está muitos desvios acima da média. Com teto infinito, é a normal truncada só por baixo.
    """
    a = (piso - media) / desvio
    b = (teto - media) / desvio
    log_cauda_a = special.log_ndtr(-a)  # log P(Z > a)
    log_cauda_b = special.log_ndtr(-b)  # log P(Z > b); -inf quando não há teto
    u = gerador.uniform(size=np.shape(a))
    # P(Z > z) sorteada uniformemente entre P(Z > b) e P(Z > a).
    with np.errstate(divide="ignore"):
        log_faixa = log_cauda_a + np.log1p(-np.exp(log_cauda_b - log_cauda_a))
        log_p = np.logaddexp(log_cauda_b, np.log(u) + log_faixa)
    z = -special.ndtri_exp(log_p)
    return media + desvio * np.clip(z, a, b)


def _log_alvo_tau(tau: np.ndarray, priori: Priori, media: np.ndarray, centro: np.ndarray, variancia: np.ndarray, com_dado: np.ndarray) -> np.ndarray:
    """log p(tau | mu, sigma, y) + log(tau), com theta integrado fora. O log(tau) é o jacobiano do passeio em log(tau).

    media: (C, J) média observada de cada grupo; centro: (C, J) mu do bloco; variancia: (C, J) sigma²/n.
    """
    total = tau[:, None] ** 2 + variancia
    termos = np.where(com_dado, -0.5 * np.log(total) - 0.5 * (media - centro) ** 2 / total, 0.0)
    return priori.log_densidade(tau) + np.log(tau) + termos.sum(axis=1)


def amostrar(
    dados: Estrutura,
    priori_tau: Priori | None = None,
    *,
    tau_fixo: float | None = None,
    sigma_fixo: np.ndarray | float | None = None,
    cadeias: int = 4,
    iteracoes: int = 2000,
    aquecimento: int = 500,
    semente: int = 0,
    media_mu: float = 4.0,
    desvio_mu: float = 10.0,
    a_sigma: float = 0.5,
    b_sigma: float = 0.125,
) -> Posteriori:
    """Roda `cadeias` cadeias em paralelo (vetorizadas) e devolve as amostras depois do aquecimento.

    As cadeias partem de pontos espalhados de propósito, para o R-hat ter o que comparar.
    O passo do Metropolis em log(tau) é ajustado só durante o aquecimento (meta: aceitar perto de 44%).
    """
    priori_tau = priori_tau or semi_normal(1.0)
    gerador = np.random.default_rng(semente)
    y, cens, grupo = dados.y, dados.censurado, dados.grupo
    J, B, K, C = dados.grupos, dados.blocos, dados.escalas, cadeias

    pertence = np.zeros((len(y), J))  # observação -> grupo
    pertence[np.arange(len(y)), grupo] = 1.0
    grupo_no_bloco = np.zeros((J, B))  # grupo -> bloco
    grupo_no_bloco[np.arange(J), dados.bloco] = 1.0
    escala_obs = dados.escala[grupo]
    obs_na_escala = np.zeros((len(y), K))  # observação -> grupo de dispersão
    obs_na_escala[np.arange(len(y)), escala_obs] = 1.0
    n = pertence.sum(axis=0)  # posts por grupo
    com_dado = n > 0
    n_seguro = np.maximum(n, 1.0)
    n_escala = obs_na_escala.sum(axis=0)
    ha_censura = bool(cens.any())

    # Pontos de partida espalhados.
    media_inicial = (y @ pertence) / n_seguro
    desvio_inicial = max(float(np.std(y)), 0.3)
    theta = media_inicial + gerador.normal(0.0, desvio_inicial, (C, J))
    mu = (theta @ grupo_no_bloco) / np.maximum(grupo_no_bloco.sum(axis=0), 1.0) + gerador.normal(0.0, desvio_inicial, (C, B))
    if sigma_fixo is None:
        sigma = desvio_inicial * np.exp(gerador.normal(0.0, 0.5, (C, K)))
    else:
        sigma = np.broadcast_to(np.asarray(sigma_fixo, dtype=float), (C, K)).copy()
    tau = np.full(C, float(tau_fixo)) if tau_fixo is not None else np.exp(gerador.normal(0.0, 1.0, C))
    y_completo = np.broadcast_to(y, (C, len(y))).copy()
    log_passo = np.zeros(C)
    aceitos = np.zeros(C)

    guardar = {"theta": np.empty((C, iteracoes, J)), "mu": np.empty((C, iteracoes, B)), "sigma": np.empty((C, iteracoes, K)), "tau": np.empty((C, iteracoes))}

    for t in range(aquecimento + iteracoes):
        # 1. Valor final dos posts censurados (aumento de dados).
        if ha_censura:
            y_completo[:, cens] = normal_truncada(gerador, theta[:, grupo[cens]], sigma[:, escala_obs[cens]], y[cens], dados.teto[cens])

        soma = y_completo @ pertence
        media = soma / n_seguro
        variancia = sigma[:, dados.escala] ** 2 / n_seguro  # variância da média observada do grupo, dado theta

        # 2. mu | tau, sigma, y, com theta integrado: média ponderada das médias dos grupos do bloco.
        peso = np.where(com_dado, 1.0 / (tau[:, None] ** 2 + variancia), 0.0)
        precisao = peso @ grupo_no_bloco + 1.0 / desvio_mu**2
        mu = ((peso * media) @ grupo_no_bloco + media_mu / desvio_mu**2) / precisao + gerador.normal(size=(C, B)) / np.sqrt(precisao)
        centro = mu[:, dados.bloco]

        # 3. tau | mu, sigma, y, com theta integrado: Metropolis com passeio aleatório em log(tau).
        if tau_fixo is None:
            proposta = tau * np.exp(np.exp(log_passo) * gerador.normal(size=C))
            razao = _log_alvo_tau(proposta, priori_tau, media, centro, variancia, com_dado) - _log_alvo_tau(tau, priori_tau, media, centro, variancia, com_dado)
            aceita = np.log(gerador.uniform(size=C)) < razao
            tau = np.where(aceita, proposta, tau)
            if t < aquecimento:
                log_passo += (aceita - 0.44) * 4.0 / np.sqrt(t + 10.0)
            else:
                aceitos += aceita

        # 4. theta | mu, tau, sigma, y: compromisso entre a média do canal e a média do bloco.
        precisao_dado = n / sigma[:, dados.escala] ** 2
        precisao_priori = 1.0 / tau[:, None] ** 2
        precisao = precisao_dado + precisao_priori
        theta = (soma / sigma[:, dados.escala] ** 2 + centro * precisao_priori) / precisao + gerador.normal(size=(C, J)) / np.sqrt(precisao)

        # 5. sigma² | theta, y: gama-inversa com a soma dos quadrados dos resíduos.
        if sigma_fixo is None:
            residuo2 = (y_completo - theta[:, grupo]) ** 2
            forma = a_sigma + n_escala / 2.0
            taxa = b_sigma + (residuo2 @ obs_na_escala) / 2.0
            sigma = np.sqrt(taxa / gerador.gamma(forma, size=(C, K)))

        if t >= aquecimento:
            i = t - aquecimento
            guardar["theta"][:, i], guardar["mu"][:, i], guardar["sigma"][:, i], guardar["tau"][:, i] = theta, mu, sigma, tau

    aceitacao = aceitos / iteracoes if tau_fixo is None else np.full(C, np.nan)
    return Posteriori(guardar["theta"], guardar["mu"], guardar["sigma"], guardar["tau"], aceitacao, dados)


# --------------------------------------------------------------------------------------- previsão


def log_densidade_preditiva(posteriori: Posteriori, y_novo: float, grupo: int) -> float:
    """log p(y_novo | dados) para um post novo do grupo: média, sobre as amostras, da densidade normal."""
    theta = posteriori.juntas("theta")[:, grupo]
    sigma = posteriori.sigma_do_grupo()[:, grupo]
    return float(special.logsumexp(stats.norm.logpdf(y_novo, theta, sigma)) - np.log(len(theta)))


def replicar(posteriori: Posteriori, gerador: np.random.Generator, amostras: int = 1000) -> np.ndarray:
    """Conjuntos de dados simulados a partir da posteriori, um por amostra sorteada: `(amostras, N)`.

    Cada conjunto repete o desenho dos dados reais (um valor para cada post, no canal dele).
    """
    dados = posteriori.estrutura
    theta = posteriori.juntas("theta")
    sigma = posteriori.sigma_do_grupo()
    escolhidas = gerador.choice(len(theta), size=amostras, replace=amostras > len(theta))
    return gerador.normal(theta[escolhidas][:, dados.grupo], sigma[escolhidas][:, dados.grupo])


# --------------------------------------------------------------------------------------- respostas analíticas


# Nas três funções, `media` é a média observada de cada grupo e `variancia` é a variância dessa média
# (sigma²/n), tratada como conhecida. É o modelo normal-normal dos livros, com resposta em fórmula fechada.


def posteriori_conjugada(media: np.ndarray, variancia: np.ndarray, mu: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Sigma, mu e tau conhecidos: média e desvio a posteriori de cada theta.

    A média a posteriori é a média do grupo encolhida em direção a mu, com peso dado pelas precisões.
    """
    precisao_dado = 1.0 / np.asarray(variancia, dtype=float)
    precisao_priori = 1.0 / tau**2
    precisao = precisao_dado + precisao_priori
    return (precisao_dado * np.asarray(media) + precisao_priori * mu) / precisao, 1.0 / np.sqrt(precisao)


def posteriori_de_mu(media: np.ndarray, variancia: np.ndarray, tau: float, media_mu: float, desvio_mu: float) -> tuple[float, float]:
    """Sigma e tau conhecidos, mu com priori normal: média e desvio a posteriori de mu (theta integrado fora)."""
    peso = 1.0 / (tau**2 + np.asarray(variancia, dtype=float))
    precisao = peso.sum() + 1.0 / desvio_mu**2
    return float(((peso * np.asarray(media)).sum() + media_mu / desvio_mu**2) / precisao), float(1.0 / np.sqrt(precisao))


def densidade_marginal_de_tau(grade: np.ndarray, priori: Priori, media: np.ndarray, variancia: np.ndarray, media_mu: float, desvio_mu: float) -> np.ndarray:
    """Sigma conhecido, um bloco: densidade a posteriori de tau numa grade, por integração exata de mu e theta.

    Dado tau, as médias dos grupos são conjuntamente normais com média `media_mu` e covariância
    diag(tau² + variancia) + desvio_mu² (todos compartilham mu). A densidade é normalizada na grade.
    """
    media = np.asarray(media, dtype=float)
    variancia = np.asarray(variancia, dtype=float)
    log_post = np.empty(len(grade))
    for i, tau in enumerate(grade):
        covariancia = np.diag(tau**2 + variancia) + desvio_mu**2
        log_post[i] = stats.multivariate_normal.logpdf(media, np.full(len(media), media_mu), covariancia)
    log_post = log_post + priori.log_densidade(grade)
    densidade = np.exp(log_post - log_post.max())
    return densidade / np.trapezoid(densidade, grade)
