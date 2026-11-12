"""Carga e preparo dos extratos anonimizados de data/ (copiados do estudo nota-viral-cortes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DADOS = Path(__file__).resolve().parents[2] / "data"

# Um post com menos de 24 horas ainda está ganhando views: o número observado é um piso, não o valor final.
HORAS_DE_MATURIDADE = 24.0
# Com 12 horas ou mais, o grosso das views já chegou: o valor final fica entre o observado e FATOR_MAXIMO vezes
# o observado. Entre as duas coletas, o post que mais cresceu depois de 12 horas no ar cresceu 3,1 vezes; 5 é folga.
# Com menos de 12 horas não há teto: o post ainda pode ir a qualquer lugar.
HORAS_PARA_TER_TETO = 12.0
FATOR_MAXIMO = 5.0

PLATAFORMAS = ["tiktok", "youtube"]
NOME_DA_PLATAFORMA = {"tiktok": "TikTok", "youtube": "YouTube"}
NICHOS = ["Games", "Negócios", "Ciência"]
# Ordem fixa dos seis canais: plataforma por fora, nicho por dentro.
CANAIS = [f"{NOME_DA_PLATAFORMA[p]} · {n}" for p in PLATAFORMAS for n in NICHOS]


def carregar_posts(pasta: Path = DADOS) -> pd.DataFrame:
    """Uma linha por post e por coleta de métricas."""
    return pd.read_csv(pasta / "posts_metricas.csv")


def carregar_agenda(pasta: Path = DADOS) -> pd.DataFrame:
    """Posts agendados por semana e por plataforma."""
    return pd.read_csv(pasta / "agenda_semanal.csv")


def preparar(
    posts: pd.DataFrame,
    coleta: str | None = None,
    horas_de_maturidade: float = HORAS_DE_MATURIDADE,
    fator_maximo: float | None = FATOR_MAXIMO,
    horas_para_ter_teto: float = HORAS_PARA_TER_TETO,
) -> pd.DataFrame:
    """Posts de uma coleta (a mais recente, por padrão), prontos para o modelo.

    - `y` é log(1 + views): a escala em que as views se comportam bem, e o 1 resolve os posts com zero views.
    - `jovem` marca o post com menos de 24 horas no ar. Para ele, `y` é um piso do valor final
      (as views só sobem), e o modelo o trata como observação censurada.
    - `teto` é o limite superior do valor final do post jovem: `y + log(fator_maximo)` para quem já tem
      12 horas no ar, infinito para quem tem menos (ou para todos, se `fator_maximo` for None).
    - `i_canal`, `i_plataforma` e `i_nicho` são os índices na ordem fixa de `CANAIS`, `PLATAFORMAS` e `NICHOS`.
    """
    coleta = coleta or posts["coleta"].max()
    tabela = posts[posts["coleta"] == coleta].copy()
    tabela["canal"] = tabela["plataforma"].map(NOME_DA_PLATAFORMA) + " · " + tabela["nicho"]
    tabela["y"] = np.log1p(tabela["views"].astype(float))
    tabela["jovem"] = tabela["horas_no_ar"] < horas_de_maturidade
    tem_teto = tabela["jovem"] & (tabela["horas_no_ar"] >= horas_para_ter_teto) & (fator_maximo is not None)
    tabela["teto"] = np.where(tem_teto, tabela["y"] + np.log(fator_maximo or 1.0), np.inf)
    tabela["i_canal"] = tabela["canal"].map({c: i for i, c in enumerate(CANAIS)})
    tabela["i_plataforma"] = tabela["plataforma"].map({p: i for i, p in enumerate(PLATAFORMAS)})
    tabela["i_nicho"] = tabela["nicho"].map({n: i for i, n in enumerate(NICHOS)})
    colunas = ["coleta", "corte", "nicho", "plataforma", "canal", "publicado_em", "horas_no_ar", "views", "y", "jovem", "teto", "i_canal", "i_plataforma", "i_nicho"]
    return tabela[colunas].sort_values(["i_canal", "publicado_em"]).reset_index(drop=True)


def crescimento_entre_coletas(posts: pd.DataFrame) -> pd.DataFrame | None:
    """Quanto as views de um mesmo post subiram entre a penúltima e a última coleta, por idade na primeira.

    É a evidência para o tratamento do post jovem: quem já tinha 24 horas quase não muda, quem
    tinha entre 12 e 24 ainda cresce um pouco, quem tinha menos de 12 pode multiplicar.
    O fator é (views depois + 1) / (views antes + 1).
    """
    coletas = sorted(posts["coleta"].unique())
    if len(coletas) < 2:
        return None
    antes = posts[posts["coleta"] == coletas[-2]].set_index(["corte", "plataforma"])
    depois = posts[posts["coleta"] == coletas[-1]].set_index(["corte", "plataforma"])
    pares = antes[["horas_no_ar", "views"]].join(depois[["views"]].rename(columns={"views": "views_depois"}), how="inner")
    faixas = ["menos de 12 h", "12 a 24 h", "24 h ou mais"]
    pares["faixa"] = pd.cut(pares["horas_no_ar"], [-np.inf, HORAS_PARA_TER_TETO, HORAS_DE_MATURIDADE, np.inf], right=False, labels=faixas)
    pares["fator"] = (pares["views_depois"] + 1) / (pares["views"] + 1)
    resumo = pares.groupby("faixa", observed=True).agg(
        posts=("fator", "size"),
        fator_mediano=("fator", "median"),
        fator_p90=("fator", lambda v: float(np.quantile(v, 0.9))),
        fator_maximo=("fator", "max"),
        cresceram_mais_de_10_por_cento=("fator", lambda v: float((v > 1.1).mean())),
    )
    resumo.index = resumo.index.astype(str)
    resumo.attrs["horas_entre_coletas"] = float((depois["horas_no_ar"] - antes["horas_no_ar"]).dropna().median())
    return resumo
