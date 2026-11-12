"""Orquestra a análise. Cada função devolve uma tabela ou um dicionário usado no notebook e no README."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bayes_canais import bandit, diagnosticos as diag, hierarquico as hier
from bayes_canais.dados import CANAIS, NICHOS, PLATAFORMAS

PLATAFORMA_DO_CANAL = np.array([0, 0, 0, 1, 1, 1])
NICHO_DO_CANAL = np.array([0, 1, 2, 0, 1, 2])

# As formas de estimar o desempenho de cada canal. As três primeiras são as do enunciado; as duas últimas
# servem de comparação (a diferença está só entre plataformas? a estrutura de plataforma importa?).
NOME_DA_FORMA = {
    "sem_agrupamento": "Sem agrupamento (cada canal sozinho)",
    "agrupamento_total": "Agrupamento total (todos iguais)",
    "hierarquico": "Hierárquico (agrupamento parcial)",
    "por_plataforma": "Uma média por plataforma",
    "hierarquico_sem_plataforma": "Hierárquico sem a plataforma",
    "media_crua": "Média crua do canal (linha de base)",
}
FORMAS_DO_MODELO = [forma for forma in NOME_DA_FORMA if forma != "media_crua"]  # as que passam pelo amostrador
PRIORIS = {
    "semi_normal": hier.semi_normal(1.0),
    "semi_cauchy": hier.semi_cauchy(1.0),
    "uniforme": hier.uniforme(5.0),
}
PRIORI_PRINCIPAL = "semi_normal"


@dataclass(frozen=True)
class Orcamento:
    """Tamanho das cadeias. O custo por iteração quase não depende do número de cadeias (são vetorizadas)."""

    cadeias: int = 4
    iteracoes: int = 4000
    aquecimento: int = 1000


PRINCIPAL = Orcamento(4, 4000, 1000)
VALIDACAO = Orcamento(8, 500, 250)  # muitos reajustes: cadeias mais curtas, mais cadeias


@dataclass(frozen=True)
class Ajuste:
    """Uma posteriori e o mapa de cada canal para o grupo que o estima naquela forma."""

    forma: str
    posteriori: hier.Posteriori
    grupo_do_canal: np.ndarray

    def theta(self) -> np.ndarray:
        """Desempenho de cada canal, em log: `(amostras, 6)`."""
        return self.posteriori.juntas("theta")[:, self.grupo_do_canal]

    def sigma(self) -> np.ndarray:
        """Dispersão dos posts de cada canal, em log: `(amostras, 6)`."""
        return self.posteriori.sigma_do_grupo()[:, self.grupo_do_canal]


def montar(tabela: pd.DataFrame, forma: str) -> tuple[hier.Estrutura, np.ndarray, float | None]:
    """Traduz a tabela de posts para a estrutura do modelo em cada forma de estimar.

    Devolve a estrutura, o grupo que estima cada canal e o tau fixo (None quando tau é estimado).
    """
    y, censurado, teto = tabela["y"].to_numpy(), tabela["jovem"].to_numpy(), tabela["teto"].to_numpy()
    canal, plataforma = tabela["i_canal"].to_numpy(), tabela["i_plataforma"].to_numpy()
    seis, zeros = np.arange(6), np.zeros(6, dtype=int)
    if forma == "sem_agrupamento":  # média e dispersão próprias de cada canal
        return hier.Estrutura(y, censurado, canal, seis, seis, teto), seis, hier.TAU_SEM_AGRUPAMENTO
    if forma == "agrupamento_total":  # uma média e uma dispersão para tudo
        return hier.Estrutura(y, censurado, np.zeros(len(y), dtype=int), zeros[:1], zeros[:1], teto), zeros, hier.TAU_SEM_AGRUPAMENTO
    if forma == "por_plataforma":  # uma média e uma dispersão por plataforma
        return hier.Estrutura(y, censurado, plataforma, np.arange(2), np.arange(2), teto), PLATAFORMA_DO_CANAL, hier.TAU_SEM_AGRUPAMENTO
    if forma == "hierarquico":  # canais em torno da média da plataforma; dispersão por plataforma
        return hier.Estrutura(y, censurado, canal, PLATAFORMA_DO_CANAL, PLATAFORMA_DO_CANAL, teto), seis, None
    if forma == "hierarquico_sem_plataforma":  # os seis canais em torno de uma média só
        return hier.Estrutura(y, censurado, canal, zeros, PLATAFORMA_DO_CANAL, teto), seis, None
    raise ValueError(f"forma desconhecida: {forma}")


def ajustar(tabela: pd.DataFrame, forma: str, priori: hier.Priori | None = None, orcamento: Orcamento = PRINCIPAL, semente: int = 0) -> Ajuste:
    estrutura, grupo_do_canal, tau_fixo = montar(tabela, forma)
    posteriori = hier.amostrar(
        estrutura,
        priori or PRIORIS[PRIORI_PRINCIPAL],
        tau_fixo=tau_fixo,
        cadeias=orcamento.cadeias,
        iteracoes=orcamento.iteracoes,
        aquecimento=orcamento.aquecimento,
        semente=semente,
    )
    return Ajuste(forma, posteriori, grupo_do_canal)


def _intervalo(amostras: np.ndarray, eixo: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mediana e intervalo de credibilidade de 95%."""
    q = np.quantile(amostras, [0.5, 0.025, 0.975], axis=eixo)
    return q[0], q[1], q[2]


# --------------------------------------------------------------------------------------- descrição


def resumo_da_amostra(tabela: pd.DataFrame, posts: pd.DataFrame) -> dict:
    return {
        "posts": int(len(tabela)),
        "cortes": int(tabela["corte"].nunique()),
        "canais": int(tabela["canal"].nunique()),
        "posts_maduros": int((~tabela["jovem"]).sum()),
        "posts_jovens": int(tabela["jovem"].sum()),
        "posts_com_zero_views": int((tabela["views"] == 0).sum()),
        "coletas": int(posts["coleta"].nunique()),
        "coleta_usada": str(tabela["coleta"].iloc[0]),
        "primeiro_post": str(tabela["publicado_em"].min()),
        "ultimo_post": str(tabela["publicado_em"].max()),
        "posts_por_canal_minimo": int(tabela.groupby("canal").size().min()),
        "posts_por_canal_maximo": int(tabela.groupby("canal").size().max()),
    }


def media_crua_por_canal(tabela: pd.DataFrame) -> pd.DataFrame:
    """A linha de base: a média crua de views de cada canal, com todos os posts e sem olhar a idade."""
    resumo = tabela.groupby("canal").agg(
        posts=("views", "size"),
        posts_jovens=("jovem", "sum"),
        media_de_views=("views", "mean"),
        mediana_de_views=("views", "median"),
        maior_post=("views", "max"),
        desvio_do_log=("y", "std"),
    )
    resumo = resumo.reindex(CANAIS)
    # Quanto da média crua vem do maior post: mede o quanto um post só decide a ordem dos canais.
    resumo["parcela_do_maior_post"] = resumo["maior_post"] / (resumo["media_de_views"] * resumo["posts"])
    resumo["posicao_pela_media_crua"] = resumo["media_de_views"].rank(ascending=False).astype(int)
    return resumo


def lote_por_dia(tabela: pd.DataFrame) -> pd.DataFrame:
    """Mediana de views por canal e por dia de publicação: mostra se o canal é estável de um dia para o outro."""
    dia = tabela["publicado_em"].str.slice(0, 10)
    return tabela.assign(dia=dia).pivot_table(index="canal", columns="dia", values="views", aggfunc="median").reindex(CANAIS)


# --------------------------------------------------------------------------------------- estimativas


def estimativas_por_canal(ajustes: dict[str, Ajuste]) -> pd.DataFrame:
    """Desempenho de cada canal em cada forma: média em log e views do post típico, com intervalo de 95%."""
    linhas = []
    for forma, ajuste in ajustes.items():
        theta = ajuste.theta()
        mediana, inferior, superior = _intervalo(theta)
        for j, canal in enumerate(CANAIS):
            linhas.append(
                {
                    "canal": canal,
                    "forma": forma,
                    "theta": float(theta[:, j].mean()),
                    "theta_desvio": float(theta[:, j].std()),
                    "tipico": float(np.expm1(mediana[j])),
                    "tipico_inferior": float(np.expm1(inferior[j])),
                    "tipico_superior": float(np.expm1(superior[j])),
                }
            )
    return pd.DataFrame(linhas)


def encolhimento(ajustes: dict[str, Ajuste]) -> pd.DataFrame:
    """Quanto o modelo hierárquico puxa cada canal da estimativa isolada em direção à média da plataforma.

    0 = ficou onde a estimativa isolada estava; 1 = foi inteiro para a média da plataforma.
    """
    sozinho = ajustes["sem_agrupamento"].theta().mean(axis=0)
    parcial = ajustes["hierarquico"].theta().mean(axis=0)
    media_da_plataforma = ajustes["hierarquico"].posteriori.juntas("mu").mean(axis=0)[PLATAFORMA_DO_CANAL]
    largura_sozinho = np.diff(np.quantile(ajustes["sem_agrupamento"].theta(), [0.025, 0.975], axis=0), axis=0)[0]
    largura_parcial = np.diff(np.quantile(ajustes["hierarquico"].theta(), [0.025, 0.975], axis=0), axis=0)[0]
    distancia = sozinho - media_da_plataforma
    # Canal que já estava colado na média não tem para onde encolher: a razão não é informativa.
    razao = np.where(np.abs(distancia) >= 0.15, (sozinho - parcial) / np.where(distancia == 0, 1.0, distancia), np.nan)
    return pd.DataFrame(
        {
            "theta_sem_agrupamento": sozinho,
            "theta_hierarquico": parcial,
            "media_da_plataforma": media_da_plataforma,
            "encolhimento": razao,
            "largura_do_intervalo_sem_agrupamento": largura_sozinho,
            "largura_do_intervalo_hierarquico": largura_parcial,
        },
        index=pd.Index(CANAIS, name="canal"),
    )


def diagnosticos_do_ajuste(ajuste: Ajuste) -> pd.DataFrame:
    """R-hat e tamanho efetivo de amostra de cada parâmetro, com o resumo a posteriori."""
    posteriori = ajuste.posteriori
    nomes_theta = CANAIS if posteriori.theta.shape[2] == 6 else [f"grupo {j}" for j in range(posteriori.theta.shape[2])]
    series = {f"theta · {nome}": posteriori.theta[:, :, j] for j, nome in enumerate(nomes_theta)}
    series |= {f"mu · bloco {b}": posteriori.mu[:, :, b] for b in range(posteriori.mu.shape[2])}
    series |= {f"sigma · grupo de dispersão {k}": posteriori.sigma[:, :, k] for k in range(posteriori.sigma.shape[2])}
    if not np.isnan(posteriori.aceitacao_tau).all():
        series["tau"] = posteriori.tau
    linhas = {}
    for nome, cadeias in series.items():
        mediana, inferior, superior = _intervalo(cadeias.ravel())
        linhas[nome] = {
            "media": float(cadeias.mean()),
            "mediana": float(mediana),
            "inferior": float(inferior),
            "superior": float(superior),
            "rhat": diag.rhat(cadeias),
            "amostra_efetiva": diag.tamanho_efetivo(cadeias),
        }
    return pd.DataFrame(linhas).T


def pior_diagnostico(tabela: pd.DataFrame) -> dict:
    return {"maior_rhat": float(tabela["rhat"].max()), "menor_amostra_efetiva": float(tabela["amostra_efetiva"].min())}


# --------------------------------------------------------------------------------------- decisão a partir da posteriori


def probabilidade_de_ser_o_melhor(theta: np.ndarray) -> np.ndarray:
    """Parcela das amostras a posteriori em que cada canal tem o maior desempenho."""
    return np.bincount(theta.argmax(axis=1), minlength=theta.shape[1]) / len(theta)


def decisao(ajuste: Ajuste) -> dict:
    """O que a posteriori diz para a decisão: quem é o melhor, com que certeza, e quanto custa errar."""
    theta = ajuste.theta()
    tipico = np.expm1(theta)
    p_melhor = probabilidade_de_ser_o_melhor(theta)
    perda = (tipico.max(axis=1, keepdims=True) - tipico).mean(axis=0)  # views do post típico perdidas, em média, apostando em cada canal

    mu = ajuste.posteriori.juntas("mu")
    razao = np.exp(mu[:, 0] - mu[:, 1])  # post típico do TikTok sobre o do YouTube
    r_med, r_inf, r_sup = _intervalo(razao)

    # Dentro de cada plataforma: quanto o melhor canal rende a mais que o pior.
    dentro = {}
    for p, nome in enumerate(PLATAFORMAS):
        do_bloco = theta[:, PLATAFORMA_DO_CANAL == p]
        fator = np.exp(do_bloco.max(axis=1) - do_bloco.min(axis=1))
        med, inf, sup = _intervalo(fator)
        dentro[nome] = {
            "fator_melhor_sobre_pior": float(med),
            "inferior": float(inf),
            "superior": float(sup),
            "p_diferenca_menor_que_25_por_cento": float((fator < 1.25).mean()),
            "p_melhor_dentro_da_plataforma": [float(v) for v in probabilidade_de_ser_o_melhor(do_bloco)],
        }

    # Se a escolha real é o nicho (o mesmo corte vai para as duas plataformas): soma do post típico nas duas.
    por_nicho = np.stack([tipico[:, NICHO_DO_CANAL == n].sum(axis=1) for n in range(len(NICHOS))], axis=1)
    return {
        "p_melhor": {canal: float(p) for canal, p in zip(CANAIS, p_melhor)},
        "p_melhor_e_do_tiktok": float(p_melhor[PLATAFORMA_DO_CANAL == 0].sum()),
        "lider": CANAIS[int(p_melhor.argmax())],
        "p_do_lider": float(p_melhor.max()),
        "perda_esperada_por_post": {canal: float(v) for canal, v in zip(CANAIS, perda)},
        "tiktok_sobre_youtube": {"fator": float(r_med), "inferior": float(r_inf), "superior": float(r_sup), "p_tiktok_maior": float((razao > 1).mean())},
        "dentro_da_plataforma": dentro,
        "p_melhor_nicho": {nicho: float(p) for nicho, p in zip(NICHOS, probabilidade_de_ser_o_melhor(por_nicho))},
    }


def media_de_views_no_modelo(ajuste: Ajuste) -> dict:
    """A média de views que o modelo log-normal implica, exp(theta + sigma²/2) - 1, e quem seria o melhor por ela.

    Entra só como alerta: com sigma grande, essa média é quase toda cauda extrapolada.
    """
    media = np.exp(ajuste.theta() + ajuste.sigma() ** 2 / 2) - 1
    mediana, inferior, superior = _intervalo(media)
    p_melhor = probabilidade_de_ser_o_melhor(media)
    return {
        canal: {"media_implicada": float(mediana[j]), "inferior": float(inferior[j]), "superior": float(superior[j]), "p_melhor": float(p_melhor[j])}
        for j, canal in enumerate(CANAIS)
    }


# --------------------------------------------------------------------------------------- sensibilidade


def sensibilidade_as_prioris(tabela: pd.DataFrame, principal: Ajuste, orcamento: Orcamento = PRINCIPAL, semente: int = 0) -> tuple[pd.DataFrame, dict[str, Ajuste]]:
    """Repete o modelo hierárquico com cada priori para tau e compara tau, as estimativas e a decisão."""
    gerador = np.random.default_rng(semente)
    ajustes, linhas = {}, {}
    for chave, priori in PRIORIS.items():
        ajuste = principal if chave == PRIORI_PRINCIPAL else ajustar(tabela, "hierarquico", priori, orcamento, semente)
        ajustes[chave] = ajuste
        tau = ajuste.posteriori.juntas("tau")
        theta = ajuste.theta()
        tau_priori = priori.amostrar(gerador, 200_000)
        p_melhor = probabilidade_de_ser_o_melhor(theta)
        mediana, inferior, superior = _intervalo(tau)
        linha = {
            "priori": priori.nome,
            "tau_priori_mediana": float(np.median(tau_priori)),
            "tau_priori_superior": float(np.quantile(tau_priori, 0.975)),
            "tau_mediana": float(mediana),
            "tau_inferior": float(inferior),
            "tau_superior": float(superior),
            "p_tau_menor_que_0,1": float((tau < 0.1).mean()),
            "p_tau_maior_que_0,5": float((tau > 0.5).mean()),
            "rhat_de_tau": diag.rhat(ajuste.posteriori.tau),
            "amostra_efetiva_de_tau": diag.tamanho_efetivo(ajuste.posteriori.tau),
            "aceitacao_do_metropolis": float(ajuste.posteriori.aceitacao_tau.mean()),
            "lider": CANAIS[int(p_melhor.argmax())],
            "p_melhor_e_do_tiktok": float(p_melhor[PLATAFORMA_DO_CANAL == 0].sum()),
        }
        linha |= {f"p_melhor · {canal}": float(p) for canal, p in zip(CANAIS, p_melhor)}
        linha |= {f"tipico · {canal}": float(v) for canal, v in zip(CANAIS, np.expm1(np.median(theta, axis=0)))}
        linhas[chave] = linha
    return pd.DataFrame(linhas).T, ajustes


def sensibilidade_aos_posts_jovens(ajustes: dict[str, Ajuste]) -> pd.DataFrame:
    """O mesmo modelo hierárquico com cada tratamento dos posts jovens (o nome do tratamento é a chave)."""
    linhas = {}
    for nome, ajuste in ajustes.items():
        theta = ajuste.theta()
        p_melhor = probabilidade_de_ser_o_melhor(theta)
        tau = ajuste.posteriori.juntas("tau")
        sigma = ajuste.posteriori.juntas("sigma")
        linha = {
            "posts": int(len(ajuste.posteriori.estrutura.y)),
            "tau_mediana": float(np.median(tau)),
            "sigma_tiktok": float(np.median(sigma[:, 0])),
            "sigma_youtube": float(np.median(sigma[:, 1])),
            "tiktok_sobre_youtube": float(np.median(np.exp(ajuste.posteriori.juntas("mu")[:, 0] - ajuste.posteriori.juntas("mu")[:, 1]))),
            "lider": CANAIS[int(p_melhor.argmax())],
            "p_melhor_e_do_tiktok": float(p_melhor[PLATAFORMA_DO_CANAL == 0].sum()),
        }
        linha |= {f"p_melhor · {canal}": float(p) for canal, p in zip(CANAIS, p_melhor)}
        linha |= {f"tipico · {canal}": float(v) for canal, v in zip(CANAIS, np.expm1(np.median(theta, axis=0)))}
        linhas[nome] = linha
    return pd.DataFrame(linhas).T


# --------------------------------------------------------------------------------------- checagem preditiva


def _estatisticas_da_checagem(y: np.ndarray, plataforma: np.ndarray) -> dict[str, np.ndarray]:
    """Estatísticas de dispersão e de cauda, por plataforma. `y` pode ser `(N,)` ou `(réplicas, N)`."""
    y = np.atleast_2d(y)
    tiktok, youtube = y[:, plataforma == 0], y[:, plataforma == 1]
    return {
        "TikTok: desvio do log das views": tiktok.std(axis=1, ddof=1),
        "TikTok: menor post (views)": np.expm1(tiktok.min(axis=1)),
        "TikTok: maior post (views)": np.expm1(tiktok.max(axis=1)),
        "YouTube: desvio do log das views": youtube.std(axis=1, ddof=1),
        "YouTube: parcela com até 5 views": (np.expm1(youtube) <= 5.5).mean(axis=1),
        "YouTube: maior post (views)": np.expm1(youtube.max(axis=1)),
    }


def checagem_preditiva(ajuste: Ajuste, tabela: pd.DataFrame, replicas: int = 4000, semente: int = 0) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """O modelo, depois de ajustado, gera dados parecidos com os reais?

    Simula conjuntos de dados a partir da posteriori e compara dispersão e cauda com o observado,
    usando só os posts maduros (os únicos com valor final conhecido).
    O p preditivo é a parcela das réplicas com estatística maior ou igual à observada: perto de 0
    ou de 1, o modelo não reproduz aquele aspecto.
    """
    gerador = np.random.default_rng(semente)
    maduro = ~tabela["jovem"].to_numpy()
    plataforma = tabela["i_plataforma"].to_numpy()[maduro]
    replicado = hier.replicar(ajuste.posteriori, gerador, replicas)[:, maduro]
    # Views são inteiras e não negativas: a réplica é arredondada na escala das views antes de medir.
    replicado = np.log1p(np.maximum(np.round(np.expm1(replicado)), 0.0))
    observado = _estatisticas_da_checagem(tabela["y"].to_numpy()[maduro], plataforma)
    simulado = _estatisticas_da_checagem(replicado, plataforma)
    linhas = {}
    for nome, valores in simulado.items():
        mediana, inferior, superior = _intervalo(valores)
        linhas[nome] = {
            "observado": float(observado[nome][0]),
            "replicas_mediana": float(mediana),
            "replicas_inferior": float(inferior),
            "replicas_superior": float(superior),
            "p_preditivo": float((valores >= observado[nome][0]).mean()),
        }
    return pd.DataFrame(linhas).T, simulado


# --------------------------------------------------------------------------------------- validação


def _prever(ajuste: Ajuste, teste: pd.DataFrame) -> list[dict]:
    theta = ajuste.theta()
    previsto = theta.mean(axis=0)
    linhas = []
    for _, post in teste.iterrows():
        j = int(post["i_canal"])
        grupo = int(ajuste.grupo_do_canal[j])
        linhas.append(
            {
                "corte": post["corte"],
                "canal": post["canal"],
                "forma": ajuste.forma,
                "y": float(post["y"]),
                "previsto": float(previsto[j]),
                "erro_absoluto": float(abs(post["y"] - previsto[j])),
                "log_densidade": hier.log_densidade_preditiva(ajuste.posteriori, float(post["y"]), grupo),
            }
        )
    return linhas


def _prever_pela_media_crua(treino: pd.DataFrame, teste: pd.DataFrame) -> list[dict]:
    """A linha de base: a média do log das views dos posts do canal, tomando o post jovem pelo valor de hoje.

    A densidade preditiva é uma normal com a média e o desvio amostrais do canal.
    """
    por_canal = treino.groupby("i_canal")["y"].agg(["mean", "std"])
    desvio_geral = float(treino["y"].std())
    linhas = []
    for _, post in teste.iterrows():
        media = float(por_canal.loc[post["i_canal"], "mean"])
        desvio = float(por_canal.loc[post["i_canal"], "std"])
        desvio = desvio if np.isfinite(desvio) and desvio > 0 else desvio_geral
        linhas.append(
            {
                "corte": post["corte"],
                "canal": post["canal"],
                "forma": "media_crua",
                "y": float(post["y"]),
                "previsto": media,
                "erro_absoluto": float(abs(post["y"] - media)),
                "log_densidade": float(-0.5 * np.log(2 * np.pi * desvio**2) - 0.5 * ((post["y"] - media) / desvio) ** 2),
            }
        )
    return linhas


def _prever_com(forma: str, treino: pd.DataFrame, teste: pd.DataFrame, orcamento: Orcamento, semente: int) -> list[dict]:
    if forma == "media_crua":
        return _prever_pela_media_crua(treino, teste)
    return _prever(ajustar(treino, forma, orcamento=orcamento, semente=semente), teste)


def validacao_cruzada(tabela: pd.DataFrame, formas: list[str], orcamento: Orcamento = VALIDACAO, semente: int = 0) -> pd.DataFrame:
    """Deixa um corte de fora por vez (os posts dele nas duas plataformas) e prevê os posts maduros desse corte.

    O corte inteiro sai porque o mesmo corte aparece no TikTok e no YouTube. Só post maduro é
    previsto: é o único com valor final conhecido. Os posts jovens dos outros cortes ficam no treino, como piso.
    """
    linhas = []
    cortes = tabela.loc[~tabela["jovem"], "corte"].unique()
    for i, corte in enumerate(cortes):
        fora = tabela["corte"] == corte
        treino, teste = tabela[~fora], tabela[fora & ~tabela["jovem"]]
        for forma in formas:
            linhas += _prever_com(forma, treino, teste, orcamento, semente + i)
    return pd.DataFrame(linhas)


def validacao_no_tempo(tabela: pd.DataFrame, formas: list[str], orcamento: Orcamento = VALIDACAO, semente: int = 0) -> tuple[pd.DataFrame, dict]:
    """Treina com a primeira metade dos posts maduros (por data de publicação) e prevê a segunda metade.

    É a situação em que a amostra por canal é ainda menor do que hoje, e sem olhar o futuro:
    nada publicado depois do ponto de corte entra no treino, nem como piso.
    """
    maduros = tabela[~tabela["jovem"]].sort_values("publicado_em")
    momentos = sorted(maduros["publicado_em"].unique())
    acumulado = np.array([(maduros["publicado_em"] <= m).mean() for m in momentos])
    limite = momentos[int(np.argmax(acumulado >= 0.5))]
    treino, teste = maduros[maduros["publicado_em"] <= limite], maduros[maduros["publicado_em"] > limite]
    linhas = []
    for forma in formas:
        linhas += _prever_com(forma, treino, teste, orcamento, semente)
    info = {
        "treina_ate": str(limite),
        "posts_de_treino": int(len(treino)),
        "posts_de_teste": int(len(teste)),
        "posts_de_treino_por_canal_minimo": int(treino.groupby("canal").size().min()),
        "posts_de_treino_por_canal_maximo": int(treino.groupby("canal").size().max()),
    }
    return pd.DataFrame(linhas), info


def resumo_da_validacao(previsoes: pd.DataFrame, referencia: str = "hierarquico", repeticoes: int = 4000, semente: int = 0) -> pd.DataFrame:
    """Erro preditivo de cada forma e a diferença para a referência, com intervalo de 95% por bootstrap de cortes.

    - erro absoluto médio em log (exp dele = por quanto a previsão erra, como fator);
    - log da densidade preditiva médio (maior é melhor; pune também a incerteza mal calibrada).
    O bootstrap sorteia cortes inteiros: os dois posts de um corte não são independentes.
    """
    gerador = np.random.default_rng(semente)
    previsoes = previsoes.assign(post=previsoes["corte"] + " · " + previsoes["canal"])
    erro = previsoes.pivot(index="post", columns="forma", values="erro_absoluto")
    lpd = previsoes.pivot(index="post", columns="forma", values="log_densidade")
    corte_do_post = previsoes.drop_duplicates("post").set_index("post")["corte"].reindex(erro.index)
    cortes = corte_do_post.unique()
    indices = {c: np.flatnonzero(corte_do_post.to_numpy() == c) for c in cortes}

    sorteios = []
    for _ in range(repeticoes):
        sorteio = np.concatenate([indices[c] for c in gerador.choice(cortes, size=len(cortes), replace=True)])
        sorteios.append((erro.to_numpy()[sorteio].mean(axis=0), lpd.to_numpy()[sorteio].mean(axis=0)))
    erros_boot = pd.DataFrame([s[0] for s in sorteios], columns=erro.columns)
    lpd_boot = pd.DataFrame([s[1] for s in sorteios], columns=lpd.columns)

    linhas = {}
    for forma in erro.columns:
        d_erro = erros_boot[forma] - erros_boot[referencia]
        d_lpd = lpd_boot[forma] - lpd_boot[referencia]
        linhas[forma] = {
            "posts_previstos": int(len(erro)),
            "erro_absoluto_medio": float(erro[forma].mean()),
            "erro_inferior": float(erros_boot[forma].quantile(0.025)),
            "erro_superior": float(erros_boot[forma].quantile(0.975)),
            "fator_de_erro": float(np.exp(erro[forma].mean())),
            "diferenca_de_erro_para_o_hierarquico": float(erro[forma].mean() - erro[referencia].mean()),
            "diferenca_de_erro_inferior": float(d_erro.quantile(0.025)),
            "diferenca_de_erro_superior": float(d_erro.quantile(0.975)),
            "log_densidade_media": float(lpd[forma].mean()),
            "diferenca_de_log_densidade_para_o_hierarquico": float(lpd[forma].mean() - lpd[referencia].mean()),
            "diferenca_de_log_densidade_inferior": float(d_lpd.quantile(0.025)),
            "diferenca_de_log_densidade_superior": float(d_lpd.quantile(0.975)),
        }
    return pd.DataFrame(linhas).T


# --------------------------------------------------------------------------------------- decisão sequencial


def cenarios(ajuste: Ajuste, mundos: int, semente: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mundos plausíveis: cada um é uma amostra da posteriori (desempenho e dispersão de cada canal)."""
    gerador = np.random.default_rng(semente)
    theta, sigma = ajuste.theta(), ajuste.sigma()
    escolhidos = gerador.choice(len(theta), size=mundos, replace=False)
    return theta[escolhidos], sigma[escolhidos]


def horizonte_da_agenda(agenda: pd.DataFrame, coleta: str, semanas: int = 4) -> dict:
    """Quantos posts a agenda programa nas `semanas` semanas inteiras depois da coleta.

    O horizonte da simulação é esse total, arredondado para um múltiplo de seis (posts iguais por canal).
    """
    futuras = agenda[agenda["semana"] > coleta].head(semanas)
    total = int((futuras["tiktok"] + futuras["youtube"]).sum())
    return {"semanas": semanas, "de": str(futuras["semana"].iloc[0]), "posts_na_agenda": total, "horizonte": int(round(total / 6) * 6)}


def simulacao_de_alocacao(
    ajuste: Ajuste, horizonte: int = 168, mundos: int = 2000, posts_iniciais: tuple[int, ...] = (1, 2, 3, 5, 10, 15), semente: int = 0
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compara repartição igual, tudo no líder depois de N posts por canal e amostragem de Thompson.

    Devolve o resumo por política e, para as figuras, o arrependimento acumulado de cada política
    (`(mundos, horizonte)`). "Canal ruim" é terminar num canal cujo post típico rende menos de 80% do melhor.
    """
    theta, sigma = cenarios(ajuste, mundos, semente)
    tipico = np.expm1(theta)
    melhor = theta.argmax(axis=1)
    ideal = tipico.max(axis=1) * horizonte
    politicas = {"igual": bandit.ReparticaoIgual(), "thompson": bandit.Thompson()}
    politicas |= {f"lider_{n}": bandit.ApostarNoLider(n) for n in posts_iniciais if n * theta.shape[1] < horizonte}

    linhas, curvas = {}, {}
    linha_dos_mundos = np.arange(mundos)
    for i, (chave, politica) in enumerate(politicas.items()):
        escolhas = bandit.simular(politica, theta, sigma, horizonte, semente + 1 + i)
        acumulado = bandit.arrependimento(escolhas, theta)
        curvas[chave] = acumulado
        final = acumulado[:, -1]
        linhas[chave] = {
            "politica": politica.nome,
            "posts_por_canal_antes_de_decidir": getattr(politica, "posts_por_canal", np.nan),
            "arrependimento_medio": float(final.mean()),
            "arrependimento_p10": float(np.quantile(final, 0.10)),
            "arrependimento_mediano": float(np.median(final)),
            "arrependimento_p90": float(np.quantile(final, 0.90)),
            "parcela_perdida_do_ideal": float(final.sum() / ideal.sum()),
            "parcela_dos_posts_no_melhor_canal": float((escolhas == melhor[:, None]).mean()),
            "parcela_dos_posts_no_tiktok": float((PLATAFORMA_DO_CANAL[escolhas.astype(int)] == 0).mean()),
        }
        if chave != "igual":  # a repartição igual não "termina" em canal nenhum
            terminou = bandit.canal_final(escolhas, theta.shape[1])
            linhas[chave] |= {
                "p_termina_fora_do_melhor_canal": float((terminou != melhor).mean()),
                "p_termina_em_canal_ruim": float((tipico[linha_dos_mundos, terminou] < 0.8 * tipico.max(axis=1)).mean()),
                "p_termina_na_plataforma_errada": float((PLATAFORMA_DO_CANAL[terminou] != PLATAFORMA_DO_CANAL[melhor]).mean()),
            }
    resumo = pd.DataFrame(linhas).T
    for chave in curvas:  # comparações pareadas, mundo a mundo
        resumo.loc[chave, "p_perde_menos_que_a_reparticao_igual"] = float((curvas[chave][:, -1] < curvas["igual"][:, -1]).mean())
        resumo.loc[chave, "p_perde_menos_que_thompson"] = float((curvas[chave][:, -1] < curvas["thompson"][:, -1]).mean())
    return resumo, curvas
