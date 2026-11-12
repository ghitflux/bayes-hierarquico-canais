"""Figuras do estudo. Azul é o método principal (modelo hierárquico, Thompson); laranja, a linha de base
ou a regra atual (cada canal sozinho, repartição igual); verde-água, a alternativa (tudo no líder)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from scipy import stats

from bayes_canais import hierarquico as hier
from bayes_canais.analise import NOME_DA_FORMA, Ajuste
from bayes_canais.dados import CANAIS
from bayes_canais.estilo import APAGADO, AQUA, AZUL, ESTILO, LARANJA, SUPERFICIE, TINTA, TINTA_SECUNDARIA, milhar, nova_figura, numero, percentual

AZUL_CLARO = "#86b6ef"


def _paineis(titulo: str, subtitulo: str, linhas: int, colunas: int, altura: float, largura: float = 8.2, **ajuste) -> tuple[Figure, np.ndarray]:
    """Figura com vários painéis e o mesmo cabeçalho (título = conclusão, subtítulo = o que é medido)."""
    fig, eixos = plt.subplots(linhas, colunas, figsize=(largura, altura), dpi=160, squeeze=False)
    fig.subplots_adjust(**({"top": 0.78, "left": 0.08, "right": 0.97, "bottom": 0.13, "wspace": 0.28, "hspace": 0.75} | ajuste))
    fig.text(0.02, 0.965, titulo, fontsize=13, fontweight="bold", color=TINTA, va="top")
    fig.text(0.02, 0.965 - 0.30 / altura, subtitulo, fontsize=9.5, color=TINTA_SECUNDARIA, va="top")
    return fig, eixos


def _mil(valor: float) -> str:
    return f"{numero(valor / 1000, 0)} mil" if valor >= 1000 else milhar(valor)


def arrependimento(curvas: dict[str, np.ndarray], resumo: pd.DataFrame) -> Figure:
    """Arrependimento acumulado médio de cada política ao longo dos posts, com a faixa de 80% do Thompson."""
    with plt.rc_context(ESTILO):
        horizonte = curvas["igual"].shape[1]
        # Um ponto a cada rodada completa pelos seis canais (no meio da rodada o rodízio faz a curva serrilhar).
        rodada = len(CANAIS)
        posts = np.concatenate([[0], np.arange(rodada, horizonte + 1, rodada)])

        def nos_pontos(valores: np.ndarray) -> np.ndarray:
            return np.concatenate([[0.0], valores[posts[1:] - 1]])

        fig, ax = nova_figura(
            f"Repartir igual perde {percentual(resumo.loc['igual', 'parcela_perdida_do_ideal'])} das views possíveis; "
            f"a amostragem de Thompson, {percentual(resumo.loc['thompson', 'parcela_perdida_do_ideal'])}",
            f"Views do post típico perdidas, acumuladas em {horizonte} posts, em relação a publicar sempre no melhor canal.\n"
            f"Média de {milhar(curvas['igual'].shape[0])} mundos sorteados da posteriori; a faixa cobre 80% dos mundos para o Thompson.",
            altura=4.9,
        )
        fig.subplots_adjust(top=0.76, right=0.76, left=0.10)
        series = [
            ("igual", "Repartição igual\n(regra atual)", LARANJA),
            ("lider_10", "Tudo no líder depois\nde 10 posts por canal", APAGADO),
            ("lider_3", "Tudo no líder depois\nde 3 posts por canal", AQUA),
            ("thompson", "Amostragem\nde Thompson", AZUL),
        ]
        faixa = np.quantile(curvas["thompson"], [0.1, 0.9], axis=0)
        ax.fill_between(posts, nos_pontos(faixa[0]), nos_pontos(faixa[1]), color=AZUL, alpha=0.14, linewidth=0, zorder=1)
        finais = {chave: curvas[chave].mean(axis=0)[-1] for chave, _, _ in series}
        # Rótulos diretos na ponta; afastados quando duas linhas terminam perto uma da outra.
        ordem = sorted(finais, key=finais.get)
        posicao, minimo = {}, 0.115 * max(finais.values())
        for i, chave in enumerate(ordem):
            posicao[chave] = finais[chave] if i == 0 else max(finais[chave], posicao[ordem[i - 1]] + minimo)
        for chave, rotulo, cor in series:
            media = curvas[chave].mean(axis=0)
            ax.plot(posts, nos_pontos(media), color=cor, zorder=3)
            ax.annotate(f"{rotulo}: {_mil(media[-1])}", (horizonte, posicao[chave]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=TINTA_SECUNDARIA, linespacing=1.15)
        ax.set_xlim(0, horizonte)
        ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _mil(v) if v else "0"))
        ax.set_xlabel("posts publicados")
        ax.set_ylabel("views perdidas (acumulado)")
        return fig


def decidir_cedo(resumo: pd.DataFrame) -> Figure:
    """Dois painéis: a chance de terminar num canal ruim e a parcela perdida, conforme quantos posts se espera antes de decidir."""
    with plt.rc_context(ESTILO):
        lider = resumo[resumo.index.str.startswith("lider_")].sort_values("posts_por_canal_antes_de_decidir")
        n = lider["posts_por_canal_antes_de_decidir"].astype(int).to_numpy()
        tres = lider.loc["lider_3"]
        fig, eixos = _paineis(
            f"Com 3 posts por canal, apostar tudo no líder dá num canal ruim em {percentual(tres['p_termina_em_canal_ruim'])} dos mundos",
            'Regra "tudo no líder depois de N posts por canal", em mundos sorteados da posteriori. Esperar mais reduz o risco,\n'
            "mas gasta posts nos canais fracos. Canal ruim: aquele cujo post típico rende menos de 80% do melhor.",
            1,
            2,
            altura=4.5,
            top=0.70,
            bottom=0.15,
        )
        paineis = [
            ("p_termina_em_canal_ruim", "Chance de terminar num canal ruim", eixos[0, 0]),
            ("parcela_perdida_do_ideal", "Parcela das views possíveis que se perde", eixos[0, 1]),
        ]
        for coluna, titulo, ax in paineis:
            valores = lider[coluna].astype(float).to_numpy()
            ax.plot(n, valores, "-o", color=AQUA, zorder=3)
            for x, v in zip(n, valores):
                ax.annotate(percentual(v), (x, v), xytext=(5, 7), textcoords="offset points", ha="left", fontsize=8.5, color=TINTA_SECUNDARIA)
            thompson = float(resumo.loc["thompson", coluna])
            ax.axhline(thompson, color=AZUL, linewidth=2, zorder=2)
            ax.annotate(
                f"Thompson: {percentual(thompson, 1 if thompson < 0.05 else 0)}", (n[-1], thompson), xytext=(0, 5), textcoords="offset points", ha="right", fontsize=8.5, color=TINTA_SECUNDARIA
            )
            if coluna == "parcela_perdida_do_ideal":
                igual = float(resumo.loc["igual", coluna])
                ax.axhline(igual, color=LARANJA, linewidth=2, zorder=2)
                ax.annotate(f"Repartição igual: {percentual(igual)}", (n[-1], igual), xytext=(0, 5), textcoords="offset points", ha="right", fontsize=8.5, color=TINTA_SECUNDARIA)
            ax.set_title(titulo, fontsize=10, color=TINTA, loc="left", pad=10)
            ax.set_xticks(n)
            ax.set_ylim(0, max(valores.max(), float(resumo.loc["igual", coluna]) if coluna == "parcela_perdida_do_ideal" else 0) * 1.22)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: percentual(v)))
            ax.set_xlabel("posts por canal antes de apostar tudo no líder")
        return fig


def estimativas(tabela: pd.DataFrame, crua: pd.DataFrame) -> Figure:
    """Ponto e intervalo de 95% do post típico de cada canal: sozinho contra hierárquico, com a média geral de referência."""
    with plt.rc_context(ESTILO):
        fig, ax = nova_figura(
            "Entre plataformas a diferença é clara; dentro de cada uma, os intervalos se sobrepõem",
            "Views do post típico de cada canal (mediana e intervalo de 95%): cada canal sozinho e no modelo hierárquico.\n"
            "O modelo puxa cada canal para a média da sua plataforma e estreita o intervalo.",
            altura=4.9,
        )
        fig.subplots_adjust(top=0.78, left=0.25, bottom=0.20)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        formas = [("sem_agrupamento", LARANJA, -0.16), ("hierarquico", AZUL, 0.16)]
        for forma, cor, desvio in formas:
            recorte = tabela[tabela["forma"] == forma].set_index("canal").reindex(CANAIS)
            for y, (_, linha) in enumerate(recorte.iterrows()):
                ax.plot([linha["tipico_inferior"] + 1, linha["tipico_superior"] + 1], [y + desvio] * 2, color=cor, linewidth=2, zorder=2)
                ax.plot(linha["tipico"] + 1, y + desvio, "o", color=cor, zorder=3, label=NOME_DA_FORMA[forma] if y == 0 else None)
                # O valor vai na ponta direita do intervalo, onde não cruza com a outra forma.
                ax.annotate(milhar(linha["tipico"]), (linha["tipico_superior"] + 1, y + desvio), xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5, color=TINTA_SECUNDARIA)
        total = float(tabela.loc[tabela["forma"] == "agrupamento_total", "tipico"].iloc[0])
        ax.axvline(total + 1, color=APAGADO, linewidth=1.2, zorder=1)
        ax.annotate(f"todos iguais: {milhar(total)}", (total + 1, 5.47), xytext=(5, 0), textcoords="offset points", fontsize=8.5, color=APAGADO, va="center", ha="left")
        ax.set_xscale("log")
        ax.set_xticks([1, 11, 101, 1001, 10001])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: milhar(v - 1)))
        ax.minorticks_off()
        ax.set_yticks(range(6), [f"{canal} (n = {int(crua.loc[canal, 'posts'])})" for canal in CANAIS])
        ax.set_ylim(-0.6, 5.6)
        ax.invert_yaxis()
        ax.set_xlabel("views do post típico (escala logarítmica)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.16), ncol=2, fontsize=9)
        return fig


def _densidade_positiva(amostras: np.ndarray, grade: np.ndarray) -> np.ndarray:
    """Densidade suavizada de uma variável positiva, refletindo as amostras em zero para não perder massa na borda."""
    nucleo = stats.gaussian_kde(np.concatenate([amostras, -amostras]))
    return 2 * nucleo(grade)


def prioris_de_tau(ajustes: dict[str, Ajuste], prioris: dict[str, hier.Priori], sensibilidade: pd.DataFrame) -> Figure:
    """Um painel por priori: a priori de tau (cinza) e a posteriori (azul)."""
    with plt.rc_context(ESTILO):
        superiores = sensibilidade["tau_superior"].astype(float)
        fig, eixos = _paineis(
            f"O teto de tau vai de {numero(superiores.min(), 1)} a {numero(superiores.max(), 1)} conforme a priori: seis canais informam pouco",
            "Tau é o desvio entre canais da mesma plataforma, em log. Cinza: a priori. Azul: a posteriori, com a mediana e o limite de 97,5%.",
            1,
            3,
            altura=3.9,
            top=0.70,
            bottom=0.17,
            wspace=0.16,
        )
        grade = np.linspace(0.0, 3.0, 301)
        gerador = np.random.default_rng(0)
        maximo = 0.0
        for ax, (chave, ajuste) in zip(eixos[0], ajustes.items()):
            priori = prioris[chave]
            tau = ajuste.posteriori.juntas("tau")
            densidade = _densidade_positiva(gerador.choice(tau, 4000, replace=False), grade)
            densidade_priori = np.exp(priori.log_densidade(np.linspace(1e-9, 200, 400001)))
            constante = np.trapezoid(densidade_priori, np.linspace(1e-9, 200, 400001))
            ax.plot(grade, np.exp(priori.log_densidade(grade)) / constante, color=APAGADO, zorder=2)
            ax.fill_between(grade, 0, densidade, color=AZUL, alpha=0.14, linewidth=0, zorder=1)
            ax.plot(grade, densidade, color=AZUL, zorder=3)
            maximo = max(maximo, densidade.max())
            linha = sensibilidade.loc[chave]
            ax.set_title(priori.nome, fontsize=10, color=TINTA, loc="left", pad=8)
            ax.annotate(
                f"mediana {numero(linha['tau_mediana'])}\n97,5%: {numero(linha['tau_superior'])}",
                (0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=8.5, color=TINTA_SECUNDARIA,
            )
            ax.set_xlim(0, 3)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xlabel("tau")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: numero(v, 0)))
        for i, ax in enumerate(eixos[0]):
            ax.set_ylim(0, maximo * 1.08)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: numero(v, 1)))
            if i:
                ax.set_yticklabels([])
        eixos[0, 0].set_ylabel("densidade")
        return fig


def checagem_preditiva(simulado: dict[str, np.ndarray], checagem: pd.DataFrame) -> Figure:
    """Seis painéis: a distribuição de cada estatística nos dados simulados pelo modelo e o valor observado."""
    with plt.rc_context(ESTILO):
        reprovadas = int(((checagem["p_preditivo"] < 0.025) | (checagem["p_preditivo"] > 0.975)).sum())
        fig, eixos = _paineis(
            f"O modelo acerta a dispersão do YouTube, mas falha em {reprovadas} de {len(checagem)} checagens de forma e cauda",
            "Histograma: a estatística em 4.000 conjuntos de dados simulados a partir da posteriori (posts com 24 h ou mais).\n"
            "Traço escuro: o valor observado. p é a parcela das simulações com valor maior ou igual ao observado.",
            2,
            3,
            altura=5.6,
            top=0.76,
            bottom=0.09,
            hspace=0.62,
            wspace=0.16,
        )
        for ax, (nome, valores) in zip(eixos.ravel(), simulado.items()):
            observado = float(checagem.loc[nome, "observado"])
            p = float(checagem.loc[nome, "p_preditivo"])
            reprovada = p < 0.025 or p > 0.975
            em_views = "(views)" in nome
            if em_views:  # cauda longa: o histograma é feito no log das views
                valores, observado_x = np.log10(valores + 1), np.log10(observado + 1)
                limite = (0, min(valores.max(), 6.5))
            else:
                observado_x = observado
                limite = (0, max(np.quantile(valores, 0.999), observado) * 1.08)
            contagens, _, _ = ax.hist(valores, bins=np.linspace(limite[0], limite[1], 31), color=LARANJA if reprovada else AZUL_CLARO, edgecolor=SUPERFICIE, linewidth=0.5, zorder=2)
            ax.set_ylim(0, contagens.max() * 1.5)  # folga em cima para o rótulo não cobrir as barras
            ax.plot([observado_x] * 2, [0, contagens.max() * 1.08], color=TINTA, linewidth=2, zorder=3)
            titulo = nome.replace(" (views)", "")
            ax.set_title(titulo, fontsize=9.5, color=TINTA, loc="left", pad=6)
            rotulo = milhar(observado) if em_views else percentual(observado) if "parcela" in nome else numero(observado)
            # O rótulo vai para o lado do traço com mais espaço livre.
            lado_direito = (observado_x - limite[0]) < 0.5 * (limite[1] - limite[0])
            ax.annotate(
                f"observado: {rotulo}\np = {numero(p, 3)}",
                (0.97 if lado_direito else 0.03, 0.97), xycoords="axes fraction", ha="right" if lado_direito else "left", va="top", fontsize=8.5,
                color=TINTA if reprovada else TINTA_SECUNDARIA, fontweight="bold" if reprovada else "normal",
            )
            ax.set_xlim(*limite)
            ax.set_yticks([])
            ax.grid(visible=False)
            if em_views:
                marcas = (10, 100, 1000) if limite[1] < 3.5 else (10, 1000, 100000)
                ax.set_xticks([np.log10(v + 1) for v in marcas if np.log10(v + 1) <= limite[1]])
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _mil(round(10**v - 1))))
            elif "parcela" in nome:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: percentual(v)))
            else:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: numero(v, 1)))
        return fig


def validacao(cruzada: pd.DataFrame, no_tempo: pd.DataFrame, info_do_tempo: dict) -> Figure:
    """Erro preditivo de cada forma de estimar, com intervalo de 95%, nas duas validações."""
    with plt.rc_context(ESTILO):
        fig, eixos = _paineis(
            "Na previsão, o modelo hierárquico empata com as alternativas simples",
            "Erro absoluto médio ao prever o log das views de um post que ficou de fora (menor é melhor), com intervalo de 95%\n"
            "por bootstrap de cortes. Um erro de 1,2 em log é errar as views por um fator de 3,3. Só juntar tudo erra claramente mais.",
            1,
            2,
            altura=4.4,
            top=0.72,
            left=0.335,
            bottom=0.15,
            wspace=0.10,
        )
        ordem = ["hierarquico", "por_plataforma", "media_crua", "sem_agrupamento", "hierarquico_sem_plataforma", "agrupamento_total"]
        cor = {"hierarquico": AZUL, "media_crua": LARANJA, "sem_agrupamento": LARANJA, "por_plataforma": AQUA}
        titulos = [
            (cruzada, f"Um corte de fora por vez ({int(cruzada['posts_previstos'].iloc[0])} posts)"),
            (no_tempo, f"No tempo (treina {info_do_tempo['posts_de_treino']}, prevê {info_do_tempo['posts_de_teste']})"),
        ]
        for i, (ax, (resumo, titulo)) in enumerate(zip(eixos[0], titulos)):
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", visible=True)
            for y, forma in enumerate(ordem):
                linha = resumo.loc[forma]
                c = cor.get(forma, APAGADO)
                ax.plot([linha["erro_inferior"], linha["erro_superior"]], [y, y], color=c, linewidth=2, zorder=2)
                ax.plot(linha["erro_absoluto_medio"], y, "o", color=c, zorder=3)
                ax.annotate(numero(linha["erro_absoluto_medio"]), (linha["erro_absoluto_medio"], y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5, color=TINTA_SECUNDARIA)
            ax.set_title(titulo, fontsize=10, color=TINTA, loc="left", pad=8)
            ax.set_yticks(range(len(ordem)), [NOME_DA_FORMA[f] for f in ordem] if i == 0 else [""] * len(ordem))
            ax.set_ylim(-0.7, len(ordem) - 0.5)
            ax.invert_yaxis()
            ax.set_xlim(0.5, 2.7)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: numero(v, 1)))
            ax.set_xlabel("erro absoluto médio em log")
        return fig
