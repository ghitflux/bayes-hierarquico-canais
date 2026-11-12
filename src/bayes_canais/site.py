"""Bloco `para_o_site` de reports/resultados.json: textos em três idiomas e a especificação do gráfico.

Os números entram por parâmetro, vindos da análise; aqui só há redação e formato.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def _n(valor: float, casas: int, idioma: str) -> str:
    texto = f"{valor:.{casas}f}"
    return texto if idioma == "en" else texto.replace(".", ",")


def _p(valor: float, idioma: str, casas: int = 0) -> str:
    return _n(valor * 100, casas, idioma) + "%"


MESES_EM_INGLES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _periodo(inicio: str, fim: str, idioma: str) -> str:
    """Período de publicação no formato de cada idioma (os dois dias no mesmo mês ficam compactos)."""
    a, b = date.fromisoformat(inicio[:10]), date.fromisoformat(fim[:10])
    if idioma == "en":
        if (a.year, a.month) == (b.year, b.month):
            return f"{MESES_EM_INGLES[a.month - 1]} {a.day}–{b.day}, {a.year}"
        return f"{MESES_EM_INGLES[a.month - 1]} {a.day}, {a.year} to {MESES_EM_INGLES[b.month - 1]} {b.day}, {b.year}"
    de, ate = ("de", "a") if idioma == "pt" else ("del", "al")
    if (a.year, a.month) == (b.year, b.month):
        return f"{de} {a.day} {ate} {b.day}/{b.month}/{b.year}"
    return f"{de} {a.day}/{a.month}/{a.year} {ate} {b.day}/{b.month}/{b.year}"


def bloco(amostra: dict, decisao: dict, simulacao: pd.DataFrame, curvas: dict[str, np.ndarray], lideres_por_tratamento: list[str]) -> dict:
    """Monta o bloco. `lideres_por_tratamento` são os canais líderes em cada tratamento dos posts jovens."""
    p_tiktok = decisao["p_melhor_e_do_tiktok"]
    razao = decisao["tiktok_sobre_youtube"]
    lider, p_lider = decisao["lider"], decisao["p_do_lider"]
    lider_muda = len(set(lideres_por_tratamento)) > 1
    ruim_3 = float(simulacao.loc["lider_3", "p_termina_em_canal_ruim"])
    ruim_thompson = float(simulacao.loc["thompson", "p_termina_em_canal_ruim"])
    perda_thompson = float(simulacao.loc["thompson", "parcela_perdida_do_ideal"])
    perda_igual = float(simulacao.loc["igual", "parcela_perdida_do_ideal"])
    horizonte = curvas["igual"].shape[1]
    # P(melhor é do TikTok) costuma arredondar para 100%: mostra uma casa e nunca escreve "100%".
    acima = p_tiktok > 0.999
    p_tiktok_texto = {
        "pt": ("passa de " if acima else "é de ") + _p(min(p_tiktok, 0.999), "pt", 1),
        "en": ("is above " if acima else "is ") + _p(min(p_tiktok, 0.999), "en", 1),
        "es": ("supera el " if acima else "es del ") + _p(min(p_tiktok, 0.999), "es", 1),
    }
    ruim_thompson_texto = {i: _p(ruim_thompson, i, 1 if ruim_thompson < 0.05 else 0) for i in ("pt", "en", "es")}

    muda = {
        "pt": ", e a liderança troca conforme o tratamento dos posts recentes" if lider_muda else "",
        "en": ", and the lead changes hands depending on how recent posts are handled" if lider_muda else "",
        "es": ", y el liderazgo cambia según cómo se traten los posts recientes" if lider_muda else "",
    }
    resultado = {
        "pt": (
            f"A plataforma dá para escolher: a probabilidade de o melhor canal ser do TikTok {p_tiktok_texto['pt']}, e o post típico de lá rende "
            f"{_n(razao['fator'], 0, 'pt')} vezes o do YouTube (intervalo de 95%: {_n(razao['inferior'], 0, 'pt')} a {_n(razao['superior'], 0, 'pt')}). "
            f"O canal, não: o líder ({lider}) tem só {_p(p_lider, 'pt')} de chance de ser o melhor{muda['pt']}."
        ),
        "en": (
            f"The platform can be chosen: the probability that the best channel is on TikTok {p_tiktok_texto['en']}, and a typical post there earns "
            f"{_n(razao['fator'], 0, 'en')} times a YouTube one (95% interval: {_n(razao['inferior'], 0, 'en')} to {_n(razao['superior'], 0, 'en')}). "
            f"The channel cannot: the leader ({lider}) has only a {_p(p_lider, 'en')} chance of being the best{muda['en']}."
        ),
        "es": (
            f"La plataforma sí se puede elegir: la probabilidad de que el mejor canal sea de TikTok {p_tiktok_texto['es']}, y el post típico de allí rinde "
            f"{_n(razao['fator'], 0, 'es')} veces el de YouTube (intervalo del 95%: {_n(razao['inferior'], 0, 'es')} a {_n(razao['superior'], 0, 'es')}). "
            f"El canal, no: el líder ({lider}) tiene solo un {_p(p_lider, 'es')} de probabilidad de ser el mejor{muda['es']}."
        ),
    }
    detalhe = {
        "pt": (
            f"Apostar tudo no líder depois de 3 posts por canal termina num canal ruim em {_p(ruim_3, 'pt')} dos mundos simulados; a amostragem de Thompson, em {ruim_thompson_texto['pt']}. "
            f"No gráfico, as views perdidas em {horizonte} posts: Thompson perde {_p(perda_thompson, 'pt')} do que era possível, contra {_p(perda_igual, 'pt')} da repartição igual; a faixa cobre 80% dos mundos."
        ),
        "en": (
            f"Going all in on the leader after 3 posts per channel ends up on a bad channel in {_p(ruim_3, 'en')} of simulated worlds; Thompson sampling does in {ruim_thompson_texto['en']}. "
            f"The chart shows views lost over {horizonte} posts: Thompson gives up {_p(perda_thompson, 'en')} of what was achievable, against {_p(perda_igual, 'en')} for an equal split; the band covers 80% of worlds."
        ),
        "es": (
            f"Apostarlo todo al líder tras 3 posts por canal termina en un canal malo en el {_p(ruim_3, 'es')} de los mundos simulados; el muestreo de Thompson, en el {ruim_thompson_texto['es']}. "
            f"En el gráfico, las views perdidas en {horizonte} posts: Thompson pierde el {_p(perda_thompson, 'es')} de lo posible, frente al {_p(perda_igual, 'es')} del reparto igual; la banda cubre el 80% de los mundos."
        ),
    }
    de, ate = amostra["primeiro_post"], amostra["ultimo_post"]
    dados = {
        "pt": f"{amostra['posts']} posts de {amostra['cortes']} cortes em seis canais (três nichos no TikTok e no YouTube), publicados {_periodo(de, ate, 'pt')}; operação própria, dados anonimizados.",
        "en": f"{amostra['posts']} posts from {amostra['cortes']} clips across six channels (three niches on TikTok and YouTube), published {_periodo(de, ate, 'en')}; the author's own operation, anonymized data.",
        "es": f"{amostra['posts']} posts de {amostra['cortes']} cortes en seis canales (tres nichos en TikTok y YouTube), publicados {_periodo(de, ate, 'es')}; operación propia, datos anonimizados.",
    }

    # Um ponto a cada rodada completa pelos seis canais: no meio da rodada o rodízio faz a curva serrilhar.
    canais = len(decisao["p_melhor"])
    passo = canais * max(1, int(np.ceil(horizonte / (60 * canais))))
    instantes = list(range(0, horizonte + 1, passo))
    if instantes[-1] != horizonte:
        instantes.append(horizonte)

    def pontos(valores: np.ndarray) -> list[list[float]]:
        """`valores[t - 1]` é o acumulado depois de t posts; o ponto zero é a origem."""
        return [[t, 0 if t == 0 else int(round(float(valores[t - 1])))] for t in instantes]

    series = [
        ("thompson", "principal", {"pt": "Amostragem de Thompson", "en": "Thompson sampling", "es": "Muestreo de Thompson"}),
        ("igual", "base", {"pt": "Repartição igual (regra atual)", "en": "Equal split (current rule)", "es": "Reparto igual (regla actual)"}),
        ("lider_3", "alternativa", {"pt": "Tudo no líder depois de 3 posts por canal", "en": "All in on the leader after 3 posts per channel", "es": "Todo al líder tras 3 posts por canal"}),
        ("lider_10", "neutro", {"pt": "Tudo no líder depois de 10 posts por canal", "en": "All in on the leader after 10 posts per channel", "es": "Todo al líder tras 10 posts por canal"}),
    ]
    faixa = np.quantile(curvas["thompson"], [0.1, 0.9], axis=0)
    return {
        "numero_chave": {i: f"{_p(ruim_3, i)} → {ruim_thompson_texto[i]}" for i in ("pt", "en", "es")},
        "pergunta": {
            "pt": "Com poucos posts em cada canal, em qual canal apostar, e quanto custa decidir cedo demais?",
            "en": "With only a handful of posts per channel, which one do you back, and what does deciding too early cost?",
            "es": "Con pocos posts en cada canal, ¿en cuál apostar y cuánto cuesta decidir demasiado pronto?",
        },
        "resultado": resultado,
        "detalhe": detalhe,
        "dados": dados,
        "etiquetas": ["Bayes hierárquico", "MCMC (Gibbs, Metropolis)", "R-hat e ESS", "Thompson sampling", "NumPy/SciPy"],
        "grafico": {
            "tipo": "linhas",
            "titulo": {
                "pt": "Views perdidas ao longo dos posts, por regra de alocação",
                "en": "Views lost as posts go out, by allocation rule",
                "es": "Views perdidas a lo largo de los posts, por regla de asignación",
            },
            "eixo_x": {"pt": "posts publicados", "en": "posts published", "es": "posts publicados"},
            "eixo_y": {
                "pt": "views do post típico perdidas (acumulado)",
                "en": "typical-post views lost (cumulative)",
                "es": "views del post típico perdidas (acumulado)",
            },
            "formato_x": "numero",
            "formato_y": "numero",
            "series": [{"nome": nome, "papel": papel, "pontos": pontos(curvas[chave].mean(axis=0))} for chave, papel, nome in series],
            "faixa": {"serie": 0, "inferior": pontos(faixa[0]), "superior": pontos(faixa[1])},
        },
    }
