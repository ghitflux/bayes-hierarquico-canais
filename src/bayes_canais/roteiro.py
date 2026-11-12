"""O roteiro da análise, do dado bruto à simulação. O script e o notebook chamam `executar` e usam o mesmo resultado."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from bayes_canais import analise, dados, graficos, site, site_extras

SEMENTE = 2026
TRATAMENTO_PRINCIPAL = "Jovem de 12 a 24 h entre o observado e 5x (principal)"
SO_MADUROS = "Só posts com 24 h ou mais"
SEM_TETO = "Todo jovem como piso, sem teto"


def executar(pasta: Path = dados.DADOS, semente: int = SEMENTE) -> dict:
    """Roda todos os passos e devolve tabelas, ajustes e curvas num dicionário."""
    r: dict = {}
    posts, agenda = dados.carregar_posts(pasta), dados.carregar_agenda(pasta)
    tabela = dados.preparar(posts)  # tratamento principal dos posts jovens
    so_maduros = tabela[~tabela["jovem"]].reset_index(drop=True)
    sem_teto = dados.preparar(posts, fator_maximo=None)
    r["posts"], r["tabela"] = posts, tabela

    r["amostra"] = analise.resumo_da_amostra(tabela, posts)
    r["crescimento"] = dados.crescimento_entre_coletas(posts)
    r["media_crua"] = analise.media_crua_por_canal(tabela)
    r["lote_por_dia"] = analise.lote_por_dia(tabela)

    # 1. As formas de estimar, todas com a mesma verossimilhança.
    ajustes = {forma: analise.ajustar(tabela, forma, semente=semente + i) for i, forma in enumerate(analise.FORMAS_DO_MODELO)}
    principal = ajustes["hierarquico"]
    r["ajustes"], r["principal"] = ajustes, principal
    r["estimativas"] = analise.estimativas_por_canal(ajustes)
    r["encolhimento"] = analise.encolhimento(ajustes)
    r["diagnosticos"] = {forma: analise.diagnosticos_do_ajuste(ajuste) for forma, ajuste in ajustes.items()}

    # 2. Sensibilidade: prioris de tau e tratamento dos posts jovens.
    r["prioris"], r["ajustes_por_priori"] = analise.sensibilidade_as_prioris(tabela, principal, semente=semente + 10)
    tratamentos = {
        TRATAMENTO_PRINCIPAL: principal,
        SO_MADUROS: analise.ajustar(so_maduros, "hierarquico", semente=semente + 20),
        SEM_TETO: analise.ajustar(sem_teto, "hierarquico", semente=semente + 21),
    }
    r["jovens"] = analise.sensibilidade_aos_posts_jovens(tratamentos)
    r["pior_diagnostico"] = {
        **{forma: analise.pior_diagnostico(d) for forma, d in r["diagnosticos"].items()},
        **{f"priori {chave}": analise.pior_diagnostico(analise.diagnosticos_do_ajuste(a)) for chave, a in r["ajustes_por_priori"].items() if a is not principal},
        **{f"tratamento: {nome}": analise.pior_diagnostico(analise.diagnosticos_do_ajuste(a)) for nome, a in tratamentos.items() if a is not principal},
    }

    # 3. Checagem preditiva a posteriori (no principal e, para comparar, só com os maduros).
    r["checagem"], r["checagem_simulado"] = analise.checagem_preditiva(principal, tabela, semente=semente + 30)
    r["checagem_so_maduros"], _ = analise.checagem_preditiva(tratamentos[SO_MADUROS], so_maduros, semente=semente + 31)

    # 4. Validação: um corte de fora por vez e separação no tempo.
    formas = list(analise.NOME_DA_FORMA)
    r["validacao_cruzada"] = analise.resumo_da_validacao(analise.validacao_cruzada(tabela, formas, semente=semente + 40), semente=semente + 41)
    previsoes_no_tempo, r["info_do_tempo"] = analise.validacao_no_tempo(tabela, formas, semente=semente + 42)
    r["validacao_no_tempo"] = analise.resumo_da_validacao(previsoes_no_tempo, semente=semente + 43)

    # 5. Decisão: quem é o melhor e a simulação de alocação sequencial.
    r["decisao"] = analise.decisao(principal)
    r["media_no_modelo"] = analise.media_de_views_no_modelo(principal)
    r["horizonte"] = analise.horizonte_da_agenda(agenda, r["amostra"]["coleta_usada"])
    r["simulacao"], r["curvas"] = analise.simulacao_de_alocacao(principal, horizonte=r["horizonte"]["horizonte"], semente=semente + 50)
    return r


def figuras(r: dict) -> dict[str, Figure]:
    """As seis figuras do README, na ordem em que aparecem."""
    return {
        "arrependimento": graficos.arrependimento(r["curvas"], r["simulacao"]),
        "decidir_cedo": graficos.decidir_cedo(r["simulacao"]),
        "estimativas": graficos.estimativas(r["estimativas"], r["media_crua"]),
        "prioris_de_tau": graficos.prioris_de_tau(r["ajustes_por_priori"], analise.PRIORIS, r["prioris"]),
        "checagem_preditiva": graficos.checagem_preditiva(r["checagem_simulado"], r["checagem"]),
        "validacao": graficos.validacao(r["validacao_cruzada"], r["validacao_no_tempo"], r["info_do_tempo"]),
    }


def _json(tabela: pd.DataFrame, orient: str = "index"):
    return json.loads(tabela.to_json(orient=orient, force_ascii=False))


def para_json(r: dict) -> dict:
    """O conteúdo de reports/resultados.json: todo número citado no README sai daqui."""
    tabela, crescimento, principal = r["tabela"], r["crescimento"], r["principal"]
    return {
        "amostra": r["amostra"],
        "tratamento_dos_posts_jovens": {
            "horas_de_maturidade": dados.HORAS_DE_MATURIDADE,
            "horas_para_ter_teto": dados.HORAS_PARA_TER_TETO,
            "fator_maximo": dados.FATOR_MAXIMO,
            "posts_jovens_com_teto": int(np.isfinite(tabela["teto"]).sum()),
            "posts_jovens_sem_teto": int((tabela["jovem"] & ~np.isfinite(tabela["teto"])).sum()),
            "horas_entre_coletas": None if crescimento is None else crescimento.attrs["horas_entre_coletas"],
            "crescimento_entre_coletas": None if crescimento is None else _json(crescimento),
        },
        "media_crua_por_canal": _json(r["media_crua"]),
        "mediana_de_views_por_canal_e_dia": _json(r["lote_por_dia"]),
        "orcamento_das_cadeias": {"principal": vars(analise.PRINCIPAL), "validacao": vars(analise.VALIDACAO)},
        "estimativas_por_canal": json.loads(r["estimativas"].to_json(orient="records", force_ascii=False)),
        "encolhimento": _json(r["encolhimento"]),
        "diagnosticos": {
            "hierarquico": _json(r["diagnosticos"]["hierarquico"]),
            "aceitacao_do_metropolis": float(principal.posteriori.aceitacao_tau.mean()),
            "amostras_a_posteriori": int(principal.posteriori.tau.size),
            "pior_por_ajuste": r["pior_diagnostico"],
        },
        "sensibilidade_as_prioris": _json(r["prioris"]),
        "sensibilidade_aos_posts_jovens": _json(r["jovens"]),
        "checagem_preditiva": _json(r["checagem"]),
        "checagem_preditiva_so_maduros": _json(r["checagem_so_maduros"]),
        "validacao_cruzada": _json(r["validacao_cruzada"]),
        "validacao_cruzada_desenho": {
            "cortes_deixados_de_fora": int(tabela.loc[~tabela["jovem"], "corte"].nunique()),
            "posts_previstos": int((~tabela["jovem"]).sum()),
        },
        "validacao_no_tempo": {**r["info_do_tempo"], "resumo": _json(r["validacao_no_tempo"])},
        "decisao": r["decisao"],
        "media_de_views_implicada_pelo_modelo": r["media_no_modelo"],
        "simulacao": {"horizonte": r["horizonte"], "mundos": int(r["curvas"]["igual"].shape[0]), "politicas": _json(r["simulacao"])},
        "para_o_site": {
            **site.bloco(r["amostra"], r["decisao"], r["simulacao"], r["curvas"], [str(v) for v in r["jovens"]["lider"]]),
            "graficos_extras": site_extras.graficos_extras(r, analise.PRIORIS),
        },
    }
