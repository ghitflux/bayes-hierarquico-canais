"""Forma de `para_o_site.graficos_extras`: tipos, limites, três idiomas e `leitura` presente."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bayes_canais import site_extras
from bayes_canais.dados import CANAIS

IDIOMAS = {"pt", "en", "es"}
PAPEIS = {"principal", "base", "alternativa", "neutro"}
RESULTADOS = Path(__file__).resolve().parents[1] / "reports" / "resultados.json"


def _tres_idiomas(campo) -> bool:
    return isinstance(campo, dict) and set(campo) == IDIOMAS and all(isinstance(v, str) and v.strip() for v in campo.values())


def validar(grafico: dict) -> None:
    assert grafico["tipo"] in {"linhas", "barras", "pontos"}
    for campo in ("titulo", "eixo_x", "eixo_y", "leitura"):
        assert _tres_idiomas(grafico[campo]), campo
    assert all(len(t) <= 90 for t in grafico["titulo"].values()), grafico["titulo"]
    assert grafico["formato_x"] in {"numero", "percentual", "moeda", "data"}
    assert grafico["formato_y"] in {"numero", "percentual", "moeda", "data"}
    assert 1 <= len(grafico["series"]) <= 4
    limite = 300 if grafico["tipo"] == "pontos" else 80
    for serie in grafico["series"]:
        assert _tres_idiomas(serie["nome"])
        assert serie["papel"] in PAPEIS
        assert 1 <= len(serie["pontos"]) <= limite
        for ponto in serie["pontos"]:
            assert len(ponto) == 2 and np.isfinite(ponto[1])
            if grafico["tipo"] == "barras":
                assert isinstance(ponto[0], str)
            else:
                assert np.isfinite(ponto[0])
        if "rotulos" in serie:
            assert set(serie["rotulos"]) == IDIOMAS
            assert all(len(v) == len(serie["pontos"]) for v in serie["rotulos"].values())
        if grafico["formato_y"] == "percentual":
            assert all(0 <= p[1] <= 1 for p in serie["pontos"])
    if "faixa" in grafico:
        assert grafico["tipo"] == "linhas"
        assert 0 <= grafico["faixa"]["serie"] < len(grafico["series"])


def test_graficos_sinteticos_tem_a_forma_pedida():
    simulacao = pd.DataFrame(
        {"posts_por_canal_antes_de_decidir": [np.nan, np.nan, 1, 3, 10], "p_termina_em_canal_ruim": [np.nan, 0.015, 0.4, 0.22, 0.09]},
        index=["igual", "thompson", "lider_1", "lider_3", "lider_10"],
    )
    estimativas = pd.DataFrame(
        [{"canal": c, "forma": f, "tipico": 10.0 * (i + 1)} for f in ("sem_agrupamento", "hierarquico", "agrupamento_total") for i, c in enumerate(CANAIS)]
    )
    formas = ["hierarquico", "por_plataforma", "media_crua", "sem_agrupamento", "hierarquico_sem_plataforma", "agrupamento_total"]
    validacao = pd.DataFrame({"erro_absoluto_medio": np.linspace(1.0, 2.0, 6)}, index=formas)
    estatisticas = [
        "TikTok: desvio do log das views", "TikTok: menor post (views)", "TikTok: maior post (views)",
        "YouTube: desvio do log das views", "YouTube: parcela com até 5 views", "YouTube: maior post (views)",
    ]
    checagem = pd.DataFrame({"p_preditivo": [0.997, 0.3, 0.999, 0.68, 0.009, 0.77]}, index=estatisticas)
    graficos = [
        site_extras.decidir_cedo(simulacao),
        site_extras.estimativas(estimativas),
        site_extras.validacao(validacao, validacao),
        site_extras.checagem_preditiva(checagem),
    ]
    for grafico in graficos:
        validar(grafico)
    assert "3" in graficos[3]["titulo"]["pt"]  # três checagens reprovadas no exemplo


def test_graficos_extras_do_resultados_json():
    if not RESULTADOS.exists():
        pytest.skip("rode scripts/rodar_analise.py antes")
    bloco = json.loads(RESULTADOS.read_text(encoding="utf-8"))["para_o_site"]
    extras = bloco["graficos_extras"]
    assert 3 <= len(extras) <= 5
    for grafico in extras:
        validar(grafico)
    assert all(g["titulo"]["pt"] != bloco["grafico"]["titulo"]["pt"] for g in extras)
