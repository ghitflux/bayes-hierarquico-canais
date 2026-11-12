"""Copia para data/ os extratos anonimizados do estudo `nota-viral-cortes`, de onde vêm os dados.

Uso:
    uv run python scripts/copiar_dados.py "<caminho para>/nota-viral-cortes/data"

Os extratos já estão versionados em data/ (são pequenos), então este passo só é necessário para
atualizar a leitura com uma coleta mais recente. Nada é alterado na pasta de origem. O script
confere o cabeçalho de cada arquivo e grava data/ORIGEM.json com o tamanho e o hash do que copiou.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

DESTINO = Path(__file__).resolve().parents[1] / "data"
ARQUIVOS = {
    "posts_metricas.csv": "coleta,corte,nicho,plataforma,publicado_em,hora_publicacao,horas_no_ar,views",
    "agenda_semanal.csv": "semana,tiktok,youtube",
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 2:
        raise SystemExit("Informe a pasta data/ do projeto nota-viral-cortes como único argumento.")
    origem = Path(sys.argv[1])
    DESTINO.mkdir(exist_ok=True)
    registro = {"origem": "extratos anonimizados do estudo nota-viral-cortes (github.com/ghitflux/nota-viral-cortes)", "arquivos": {}}
    for nome, cabecalho in ARQUIVOS.items():
        arquivo = origem / nome
        conteudo = arquivo.read_bytes()
        primeira_linha = conteudo.split(b"\n", 1)[0].decode("utf-8").strip()
        if not primeira_linha.startswith(cabecalho):
            raise SystemExit(f"{nome}: cabeçalho inesperado ({primeira_linha!r}). Nada foi copiado.")
        shutil.copyfile(arquivo, DESTINO / nome)
        linhas = conteudo.count(b"\n")
        registro["arquivos"][nome] = {
            "bytes": len(conteudo),
            "linhas": linhas,
            "sha256": hashlib.sha256(conteudo).hexdigest(),
        }
        print(f"{nome}: {len(conteudo)} bytes, {linhas} linhas")
    (DESTINO / "ORIGEM.json").write_text(json.dumps(registro, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Copiado para {DESTINO}")


if __name__ == "__main__":
    main()
