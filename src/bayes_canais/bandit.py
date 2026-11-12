"""Decisão sequencial: em qual canal publicar o próximo post? Políticas e simulação.

Cada "mundo" simulado tem um desempenho verdadeiro por canal (theta, em log) e uma dispersão (sigma).
Um post publicado no canal k devolve y ~ Normal(theta_k, sigma_k²), em log(1 + views). As políticas
escolhem o canal de cada post vendo só o que já publicaram. Tudo é vetorizado nos mundos:
os arrays têm formato `(mundos, canais)`.

O objetivo é o post típico (a média em log, que é a mediana das views). O arrependimento de cada
post é a diferença, em views do post típico, entre o melhor canal daquele mundo e o canal escolhido.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Historico:
    """O que a política já viu em cada canal de cada mundo."""

    n: np.ndarray  # posts publicados
    soma: np.ndarray  # soma de y
    soma2: np.ndarray  # soma de y²

    @classmethod
    def vazio(cls, mundos: int, canais: int) -> Historico:
        return cls(*(np.zeros((mundos, canais)) for _ in range(3)))

    def registrar(self, canal: np.ndarray, y: np.ndarray) -> None:
        linhas = np.arange(len(canal))
        self.n[linhas, canal] += 1
        self.soma[linhas, canal] += y
        self.soma2[linhas, canal] += y**2


class ReparticaoIgual:
    """A regra atual: um post em cada canal, em rodízio."""

    nome = "Repartição igual"

    def iniciar(self, mundos: int, canais: int) -> None:
        self.canais = canais

    def escolher(self, t: int, historico: Historico, gerador: np.random.Generator) -> np.ndarray:
        return np.full(len(historico.n), t % self.canais)


@dataclass
class ApostarNoLider:
    """A tentação comum: rodízio até cada canal ter `posts_por_canal` posts; depois, tudo no líder, para sempre.

    O líder é o canal com a maior média de log(1 + views), isto é, o melhor post típico até ali:
    o mesmo objetivo pelo qual a simulação mede o arrependimento.
    """

    posts_por_canal: int
    lider: np.ndarray | None = field(default=None, repr=False)

    @property
    def nome(self) -> str:
        return f"Tudo no líder depois de {self.posts_por_canal} posts por canal"

    def iniciar(self, mundos: int, canais: int) -> None:
        self.canais = canais
        self.lider = None

    def escolher(self, t: int, historico: Historico, gerador: np.random.Generator) -> np.ndarray:
        if t < self.posts_por_canal * self.canais:
            return np.full(len(historico.n), t % self.canais)
        if self.lider is None:
            self.lider = np.argmax(historico.soma / historico.n, axis=1)
        return self.lider


def posteriori_normal_gama_inversa(n, soma, soma2, m0: float, k0: float, a0: float, b0: float):
    """Atualização conjugada para média e variância desconhecidas (normal-gama-inversa).

    Priori: sigma² ~ Gama-Inversa(a0, b0) e mu | sigma² ~ Normal(m0, sigma²/k0).
    Devolve (m, k, a, b) da posteriori depois de `n` observações com soma `soma` e soma de quadrados `soma2`.
    """
    n = np.asarray(n, dtype=float)
    media = np.divide(soma, n, out=np.zeros_like(n), where=n > 0)
    desvios2 = np.maximum(soma2 - n * media**2, 0.0)
    k = k0 + n
    m = (k0 * m0 + soma) / k
    a = a0 + n / 2.0
    b = b0 + 0.5 * desvios2 + k0 * n * (media - m0) ** 2 / (2.0 * k)
    return m, k, a, b


@dataclass
class Thompson:
    """Amostragem de Thompson: sorteia um desempenho plausível de cada canal e publica no que saiu melhor.

    Cada canal tem a sua posteriori normal-gama-inversa, independente das outras. Canal pouco testado
    tem posteriori larga, então às vezes sorteia alto e ganha um post: a exploração sai de graça e
    diminui sozinha conforme a incerteza cai.
    """

    m0: float = 4.0
    k0: float = 0.1
    a0: float = 1.0
    b0: float = 1.0
    nome: str = "Amostragem de Thompson"

    def iniciar(self, mundos: int, canais: int) -> None:
        pass

    def escolher(self, t: int, historico: Historico, gerador: np.random.Generator) -> np.ndarray:
        m, k, a, b = posteriori_normal_gama_inversa(historico.n, historico.soma, historico.soma2, self.m0, self.k0, self.a0, self.b0)
        variancia = b / gerador.gamma(a)
        sorteio = m + np.sqrt(variancia / k) * gerador.normal(size=m.shape)
        return np.argmax(sorteio, axis=1)


def simular(politica, theta: np.ndarray, sigma: np.ndarray, horizonte: int, semente: int = 0) -> np.ndarray:
    """Roda a política por `horizonte` posts em cada mundo. Devolve o canal escolhido: `(mundos, horizonte)`."""
    gerador = np.random.default_rng(semente)
    mundos, canais = theta.shape
    politica.iniciar(mundos, canais)
    historico = Historico.vazio(mundos, canais)
    escolhas = np.empty((mundos, horizonte), dtype=np.int16)
    linhas = np.arange(mundos)
    for t in range(horizonte):
        canal = politica.escolher(t, historico, gerador)
        y = gerador.normal(theta[linhas, canal], sigma[linhas, canal])
        historico.registrar(canal, y)
        escolhas[:, t] = canal
    return escolhas


def arrependimento(escolhas: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Arrependimento acumulado, em views do post típico: `(mundos, horizonte)`.

    A cada post, soma o que o melhor canal daquele mundo renderia a mais que o canal escolhido.
    """
    tipico = np.expm1(theta)
    lacuna = tipico.max(axis=1, keepdims=True) - tipico
    return np.cumsum(np.take_along_axis(lacuna, escolhas.astype(int), axis=1), axis=1)


def canal_final(escolhas: np.ndarray, canais: int, fracao_final: float = 0.25) -> np.ndarray:
    """Em qual canal a política terminou: o mais escolhido no trecho final do horizonte."""
    trecho = escolhas[:, int(escolhas.shape[1] * (1 - fracao_final)) :].astype(int)
    contagem = np.stack([(trecho == k).sum(axis=1) for k in range(canais)], axis=1)
    return contagem.argmax(axis=1)
