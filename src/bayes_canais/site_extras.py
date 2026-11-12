"""`para_o_site.graficos_extras`: os outros achados do README como especificação em dados, para o site desenhar.

Cada gráfico equivale a uma figura de reports/figuras/ que não é o gráfico principal (o arrependimento acumulado).
Os valores saem das mesmas tabelas que geram as figuras; aqui só há redação, tradução e formato.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bayes_canais.analise import NOME_DA_FORMA, Ajuste
from bayes_canais.dados import CANAIS
from bayes_canais.hierarquico import Priori

IDIOMAS = ("pt", "en", "es")
NICHO = {"Games": {"pt": "Games", "en": "Gaming", "es": "Videojuegos"}, "Negócios": {"pt": "Negócios", "en": "Business", "es": "Negocios"}, "Ciência": {"pt": "Ciência", "en": "Science", "es": "Ciencia"}}


def _n(valor: float, casas: int, idioma: str) -> str:
    texto = f"{valor:.{casas}f}"
    return texto if idioma == "en" else texto.replace(".", ",")


def _p(valor: float, idioma: str, casas: int = 0) -> str:
    return _n(valor * 100, casas, idioma) + "%"


def _canal(canal: str, idioma: str) -> str:
    plataforma, nicho = canal.split(" · ")
    return f"{plataforma} · {NICHO[nicho][idioma]}"


def _rotulos_dos_canais() -> dict[str, list[str]]:
    return {i: [_canal(c, i) for c in CANAIS] for i in IDIOMAS}


def decidir_cedo(simulacao: pd.DataFrame) -> dict:
    """Chance de terminar num canal ruim conforme quantos posts por canal se espera antes de apostar tudo no líder."""
    lider = simulacao[simulacao.index.str.startswith("lider_")].sort_values("posts_por_canal_antes_de_decidir")
    n = lider["posts_por_canal_antes_de_decidir"].astype(int).to_list()
    ruim = lider["p_termina_em_canal_ruim"].astype(float).to_list()
    thompson = float(simulacao.loc["thompson", "p_termina_em_canal_ruim"])
    tres = float(simulacao.loc["lider_3", "p_termina_em_canal_ruim"])
    casas_t = 1 if thompson < 0.05 else 0
    return {
        "tipo": "linhas",
        "titulo": {
            "pt": f"Apostar no líder com 3 posts por canal dá num canal ruim em {_p(tres, 'pt')} das vezes",
            "en": f"Backing the leader after 3 posts per channel lands on a bad channel {_p(tres, 'en')} of the time",
            "es": f"Apostar al líder tras 3 posts por canal acaba en un canal malo el {_p(tres, 'es')} de las veces",
        },
        "eixo_x": {"pt": "posts por canal antes de apostar tudo no líder", "en": "posts per channel before going all in", "es": "posts por canal antes de apostarlo todo"},
        "eixo_y": {"pt": "chance de terminar num canal ruim", "en": "chance of ending on a bad channel", "es": "probabilidad de terminar en un canal malo"},
        "formato_x": "numero",
        "formato_y": "percentual",
        "series": [
            {"nome": {"pt": "Tudo no líder depois de N posts", "en": "All in on the leader after N posts", "es": "Todo al líder tras N posts"}, "papel": "alternativa", "pontos": [[x, round(y, 4)] for x, y in zip(n, ruim)]},
            {"nome": {"pt": "Amostragem de Thompson", "en": "Thompson sampling", "es": "Muestreo de Thompson"}, "papel": "principal", "pontos": [[n[0], round(thompson, 4)], [n[-1], round(thompson, 4)]]},
        ],
        "leitura": {
            "pt": f"Esperar mais posts reduz o risco de {_p(ruim[0], 'pt')} para {_p(ruim[-1], 'pt')}, mas gasta o mês nos canais fracos; o Thompson fica em {_p(thompson, 'pt', casas_t)} sem precisar escolher N. Canal ruim: post típico abaixo de 80% do melhor.",
            "en": f"Waiting longer cuts the risk from {_p(ruim[0], 'en')} to {_p(ruim[-1], 'en')} but burns the month on weak channels; Thompson sits at {_p(thompson, 'en', casas_t)} with no N to pick. A bad channel's typical post earns under 80% of the best one's.",
            "es": f"Esperar más posts baja el riesgo del {_p(ruim[0], 'es')} al {_p(ruim[-1], 'es')}, pero gasta el mes en los canales flojos; Thompson queda en el {_p(thompson, 'es', casas_t)} sin tener que elegir N. Canal malo: post típico por debajo del 80% del mejor.",
        },
    }


def estimativas(tabela: pd.DataFrame) -> dict:
    """Post típico de cada canal: sozinho, no modelo hierárquico e com todos iguais."""
    def valores(forma: str) -> list[float]:
        return tabela[tabela["forma"] == forma].set_index("canal").reindex(CANAIS)["tipico"].astype(float).to_list()

    sozinho, parcial, total = valores("sem_agrupamento"), valores("hierarquico"), valores("agrupamento_total")
    youtube_sozinho, youtube_parcial = sozinho[3:], parcial[3:]
    rotulos = _rotulos_dos_canais()
    nomes = {
        "hierarquico": {"pt": "Modelo hierárquico", "en": "Hierarchical model", "es": "Modelo jerárquico"},
        "sem_agrupamento": {"pt": "Cada canal sozinho", "en": "Each channel on its own", "es": "Cada canal por separado"},
        "agrupamento_total": {"pt": "Todos iguais", "en": "All channels pooled", "es": "Todos iguales"},
    }
    series = [
        {"nome": nomes["hierarquico"], "papel": "principal", "pontos": [[c, round(v, 1)] for c, v in zip(CANAIS, parcial)], "rotulos": rotulos},
        {"nome": nomes["sem_agrupamento"], "papel": "base", "pontos": [[c, round(v, 1)] for c, v in zip(CANAIS, sozinho)], "rotulos": rotulos},
        {"nome": nomes["agrupamento_total"], "papel": "neutro", "pontos": [[c, round(v, 1)] for c, v in zip(CANAIS, total)], "rotulos": rotulos},
    ]
    a, b = f"{min(youtube_sozinho):.0f} a {max(youtube_sozinho):.0f}", f"{min(youtube_parcial):.0f} a {max(youtube_parcial):.0f}"
    return {
        "tipo": "barras",
        "titulo": {
            "pt": "Entre plataformas a diferença é clara; dentro de cada uma, quase some",
            "en": "The gap between platforms is clear; within each one it nearly vanishes",
            "es": "Entre plataformas la diferencia es clara; dentro de cada una, casi desaparece",
        },
        "eixo_x": {"pt": "canal", "en": "channel", "es": "canal"},
        "eixo_y": {"pt": "views do post típico (mediana)", "en": "typical-post views (median)", "es": "views del post típico (mediana)"},
        "formato_x": "numero",
        "formato_y": "numero",
        "series": series,
        "leitura": {
            "pt": f"Sozinhos, os canais do YouTube vão de {a} views por post típico; o modelo hierárquico os aproxima da média da plataforma ({b}). Juntar tudo dá {total[0]:.0f}, que não descreve nenhuma das duas.",
            "en": f"On their own, the YouTube channels range from {a.replace(' a ', ' to ')} views per typical post; the hierarchical model pulls them toward the platform mean ({b.replace(' a ', ' to ')}). Pooling everything gives {total[0]:.0f}, which fits neither platform.",
            "es": f"Por separado, los canales de YouTube van de {a} views por post típico; el modelo jerárquico los acerca a la media de la plataforma ({b}). Juntarlo todo da {total[0]:.0f}, que no describe a ninguna de las dos.",
        },
    }


def validacao(cruzada: pd.DataFrame, no_tempo: pd.DataFrame) -> dict:
    """Erro absoluto médio em log de cada forma de estimar, nas duas validações."""
    ordem = ["hierarquico", "por_plataforma", "media_crua", "sem_agrupamento", "hierarquico_sem_plataforma", "agrupamento_total"]
    traducao = {
        "hierarquico": {"en": "Hierarchical (partial pooling)", "es": "Jerárquico (agrupamiento parcial)"},
        "por_plataforma": {"en": "One mean per platform", "es": "Una media por plataforma"},
        "media_crua": {"en": "Raw channel mean (baseline)", "es": "Media cruda del canal (referencia)"},
        "sem_agrupamento": {"en": "No pooling (each channel alone)", "es": "Sin agrupamiento (cada canal solo)"},
        "hierarquico_sem_plataforma": {"en": "Hierarchical, ignoring platform", "es": "Jerárquico sin la plataforma"},
        "agrupamento_total": {"en": "Complete pooling (all equal)", "es": "Agrupamiento total (todos iguales)"},
    }
    rotulos = {"pt": [NOME_DA_FORMA[f] for f in ordem], "en": [traducao[f]["en"] for f in ordem], "es": [traducao[f]["es"] for f in ordem]}
    hier, crua, total = (float(cruzada.loc[f, "erro_absoluto_medio"]) for f in ("hierarquico", "media_crua", "agrupamento_total"))
    return {
        "tipo": "barras",
        "titulo": {
            "pt": "Na previsão, o modelo hierárquico empata com as alternativas simples",
            "en": "When predicting, the hierarchical model ties with the simple alternatives",
            "es": "Al predecir, el modelo jerárquico empata con las alternativas simples",
        },
        "eixo_x": {"pt": "forma de estimar", "en": "estimation approach", "es": "forma de estimar"},
        "eixo_y": {"pt": "erro absoluto médio em log (menor é melhor)", "en": "mean absolute error in log (lower is better)", "es": "error absoluto medio en log (menor es mejor)"},
        "formato_x": "numero",
        "formato_y": "numero",
        "series": [
            {"nome": {"pt": "Um corte de fora por vez", "en": "Leave one clip out", "es": "Un corte fuera cada vez"}, "papel": "principal", "pontos": [[f, round(float(cruzada.loc[f, "erro_absoluto_medio"]), 3)] for f in ordem], "rotulos": rotulos},
            {"nome": {"pt": "Separação no tempo", "en": "Time-based split", "es": "Separación en el tiempo"}, "papel": "alternativa", "pontos": [[f, round(float(no_tempo.loc[f, "erro_absoluto_medio"]), 3)] for f in ordem], "rotulos": rotulos},
        ],
        "leitura": {
            "pt": f"Deixando um corte de fora, o hierárquico erra {_n(hier, 2, 'pt')} e a média crua do canal, {_n(crua, 2, 'pt')}: empate. Só juntar tudo erra claramente mais ({_n(total, 2, 'pt')}). Erro de 1,2 em log é errar as views por um fator de 3,3.",
            "en": f"Leaving one clip out, the hierarchical model errs by {_n(hier, 2, 'en')} and the raw channel mean by {_n(crua, 2, 'en')}: a tie. Only full pooling is clearly worse ({_n(total, 2, 'en')}). An error of 1.2 in log means missing views by a factor of 3.3.",
            "es": f"Dejando un corte fuera, el jerárquico se equivoca en {_n(hier, 2, 'es')} y la media cruda del canal en {_n(crua, 2, 'es')}: empate. Solo juntarlo todo falla claramente más ({_n(total, 2, 'es')}). Un error de 1,2 en log es fallar las views por un factor de 3,3.",
        },
    }


def _densidade_positiva(amostras: np.ndarray, grade: np.ndarray) -> np.ndarray:
    """Densidade suavizada de uma variável positiva, refletindo em zero (a mesma conta da figura)."""
    from scipy import stats

    nucleo = stats.gaussian_kde(np.concatenate([amostras, -amostras]))
    return 2 * nucleo(grade)


def prioris_de_tau(ajustes: dict[str, Ajuste], prioris: dict[str, Priori], sensibilidade: pd.DataFrame, semente: int = 0) -> dict:
    """Densidade a posteriori de tau sob cada priori, numa grade de 0 a 3."""
    gerador = np.random.default_rng(semente)
    grade = np.linspace(0.0, 3.0, 61)
    papeis = {"semi_normal": "principal", "semi_cauchy": "alternativa", "uniforme": "neutro"}
    series = []
    for chave, ajuste in ajustes.items():
        tau = ajuste.posteriori.juntas("tau")
        densidade = _densidade_positiva(gerador.choice(tau, 4000, replace=False), grade)
        nome = prioris[chave].nome
        series.append({
            "nome": {"pt": f"Priori {nome}", "en": f"{nome.replace('semi-', 'half-').replace('uniforme', 'uniform')} prior", "es": f"Priori {nome}"},
            "papel": papeis.get(chave, "neutro"),
            "pontos": [[round(float(x), 2), round(float(y), 4)] for x, y in zip(grade, densidade)],
        })
    superiores = sensibilidade["tau_superior"].astype(float)
    lo, hi = float(superiores.min()), float(superiores.max())
    return {
        "tipo": "linhas",
        "titulo": {
            "pt": f"O teto de tau vai de {_n(lo, 1, 'pt')} a {_n(hi, 1, 'pt')} conforme a priori: seis canais informam pouco",
            "en": f"The upper bound on tau runs from {_n(lo, 1, 'en')} to {_n(hi, 1, 'en')} by prior: six channels say little",
            "es": f"El techo de tau va de {_n(lo, 1, 'es')} a {_n(hi, 1, 'es')} según la priori: seis canales dicen poco",
        },
        "eixo_x": {"pt": "tau (desvio entre canais, em log)", "en": "tau (between-channel spread, log)", "es": "tau (desvío entre canales, en log)"},
        "eixo_y": {"pt": "densidade a posteriori", "en": "posterior density", "es": "densidad a posteriori"},
        "formato_x": "numero",
        "formato_y": "numero",
        "series": series,
        "leitura": {
            "pt": f"Com três prioris razoáveis, os dados puxam tau para perto de zero, mas não descartam valores altos: o limite de 97,5% vai de {_n(lo, 2, 'pt')} a {_n(hi, 2, 'pt')}. A decisão quase não muda.",
            "en": f"Under three reasonable priors the data pull tau toward zero but can't rule out large values: the 97.5% bound runs from {_n(lo, 2, 'en')} to {_n(hi, 2, 'en')}. The decision barely moves.",
            "es": f"Con tres prioris razonables, los datos llevan tau hacia cero, pero no descartan valores altos: el límite del 97,5% va de {_n(lo, 2, 'es')} a {_n(hi, 2, 'es')}. La decisión casi no cambia.",
        },
    }


def checagem_preditiva(checagem: pd.DataFrame) -> dict:
    """p preditivo de cada estatística: perto de 0 ou de 1, o modelo não reproduz aquele aspecto dos dados."""
    traducao = {
        "TikTok: desvio do log das views": {"en": "TikTok: spread of log views", "es": "TikTok: desvío del log de las views"},
        "TikTok: menor post (views)": {"en": "TikTok: smallest post", "es": "TikTok: post más pequeño"},
        "TikTok: maior post (views)": {"en": "TikTok: largest post", "es": "TikTok: post más grande"},
        "YouTube: desvio do log das views": {"en": "YouTube: spread of log views", "es": "YouTube: desvío del log de las views"},
        "YouTube: parcela com até 5 views": {"en": "YouTube: share with 5 views or fewer", "es": "YouTube: parte con 5 views o menos"},
        "YouTube: maior post (views)": {"en": "YouTube: largest post", "es": "YouTube: post más grande"},
    }
    nomes = list(checagem.index)
    rotulos = {"pt": [n.replace(" (views)", "") for n in nomes], "en": [traducao[n]["en"] for n in nomes], "es": [traducao[n]["es"] for n in nomes]}
    p = checagem["p_preditivo"].astype(float)
    reprovadas = int(((p < 0.025) | (p > 0.975)).sum())
    return {
        "tipo": "barras",
        "titulo": {
            "pt": f"O modelo acerta a dispersão do YouTube, mas falha em {reprovadas} de {len(p)} checagens",
            "en": f"The model gets YouTube's spread right but fails {reprovadas} of {len(p)} checks",
            "es": f"El modelo acierta la dispersión de YouTube, pero falla en {reprovadas} de {len(p)} pruebas",
        },
        "eixo_x": {"pt": "estatística (posts com 24 h ou mais)", "en": "statistic (posts 24 h or older)", "es": "estadística (posts con 24 h o más)"},
        "eixo_y": {"pt": "p preditivo a posteriori", "en": "posterior predictive p", "es": "p predictivo a posteriori"},
        "formato_x": "numero",
        "formato_y": "percentual",
        "series": [{"nome": {"pt": "p preditivo", "en": "predictive p", "es": "p predictivo"}, "papel": "principal", "pontos": [[n, round(float(v), 4)] for n, v in zip(nomes, p)], "rotulos": rotulos}],
        "leitura": {
            "pt": "Parcela de 4.000 conjuntos simulados pelo modelo com valor maior ou igual ao observado. Perto de 0% ou de 100%, o modelo não reproduz aquele aspecto: o teto do TikTok e o excesso de posts quase sem views no YouTube.",
            "en": "Share of 4,000 model-simulated datasets at or above the observed value. Near 0% or 100%, the model misses that feature: TikTok's ceiling and YouTube's excess of near-zero posts.",
            "es": "Parte de 4.000 conjuntos simulados por el modelo con valor mayor o igual al observado. Cerca del 0% o del 100%, el modelo no reproduce ese rasgo: el techo de TikTok y el exceso de posts casi sin views en YouTube.",
        },
    }


def graficos_extras(r: dict, prioris: dict[str, Priori]) -> list[dict]:
    """Os cinco gráficos extras, do achado mais importante para o menos."""
    return [
        decidir_cedo(r["simulacao"]),
        estimativas(r["estimativas"]),
        validacao(r["validacao_cruzada"], r["validacao_no_tempo"]),
        prioris_de_tau(r["ajustes_por_priori"], prioris, r["prioris"]),
        checagem_preditiva(r["checagem"]),
    ]
