"""Roda a análise completa e grava reports/resultados.json e as figuras do README.

Uso (na raiz do projeto):
    uv run python scripts/rodar_analise.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from bayes_canais import dados, roteiro

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "reports"
FIGURAS = SAIDA / "figuras"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    FIGURAS.mkdir(parents=True, exist_ok=True)

    r = roteiro.executar()
    for nome, figura in roteiro.figuras(r).items():
        figura.savefig(FIGURAS / f"{nome}.png")
        plt.close(figura)
    (SAIDA / "resultados.json").write_text(json.dumps(roteiro.para_json(r), indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    colunas = ["erro_absoluto_medio", "diferenca_de_erro_para_o_hierarquico", "diferenca_de_erro_inferior", "diferenca_de_erro_superior", "log_densidade_media"]
    print(r["amostra"])
    print(r["crescimento"].to_string() if r["crescimento"] is not None else "uma coleta só")
    print(r["media_crua"].to_string())
    print(r["estimativas"].pivot(index="canal", columns="forma", values="tipico").reindex(dados.CANAIS).to_string())
    print(r["diagnosticos"]["hierarquico"].to_string())
    print(r["prioris"].T.to_string())
    print(r["jovens"].T.to_string())
    print(r["checagem"].to_string())
    print(r["validacao_cruzada"][colunas].to_string())
    print(r["validacao_no_tempo"][colunas].to_string())
    print(r["decisao"])
    print(r["simulacao"].to_string())
    print(f"Gravado em {SAIDA / 'resultados.json'} e {FIGURAS}")
    print(f"Tempo total: {time.perf_counter() - inicio:.1f} s")


if __name__ == "__main__":
    main()
