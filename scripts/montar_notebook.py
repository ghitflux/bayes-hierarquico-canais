"""Monta analise.ipynb com nbformat. Depois de montar, execute:

    uv run jupyter nbconvert --to notebook --execute --inplace analise.ipynb

O notebook chama o mesmo roteiro do script `rodar_analise.py`, com as mesmas sementes: os números são os de reports/resultados.json.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

RAIZ = Path(__file__).resolve().parents[1]

CELULAS: list[tuple[str, str]] = [
    (
        "md",
        """# Com poucos posts em cada canal, em qual canal apostar, e quanto custa decidir cedo demais?

Uma operação de cortes de vídeo publica em seis canais: três nichos (Games, Negócios, Ciência) em duas plataformas (TikTok e YouTube). Cada canal tem de 8 a 12 posts com métrica. A média crua de cada canal engana quando a amostra é pequena; este notebook estima o desempenho de cada canal de três formas, mede o quanto a conclusão depende das escolhas do analista e simula o custo de decidir cedo.

O núcleo (amostrador de Gibbs com um passo de Metropolis, diagnósticos de convergência, amostragem de Thompson) está implementado em NumPy/SciPy no pacote `bayes_canais`, com testes contra casos de resposta conhecida em `tests/`. Aqui só se chama o pacote e se lê o resultado.""",
    ),
    (
        "code",
        """import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from bayes_canais import analise, dados, roteiro

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
pd.set_option("display.max_columns", 30)

r = roteiro.executar()   # o mesmo roteiro e as mesmas sementes de scripts/rodar_analise.py
figuras = roteiro.figuras(r)
plt.close("all")         # cada figura é mostrada na seção em que é discutida
r["amostra"]""",
    ),
    (
        "md",
        """## 1. Os dados e a linha de base

Uma linha por post, na coleta mais recente. A linha de base é o que qualquer um faria: a média de views de cada canal.""",
    ),
    (
        "code",
        """crua = r["media_crua"]
display(crua[["posts", "posts_jovens", "media_de_views", "mediana_de_views", "maior_post", "parcela_do_maior_post", "posicao_pela_media_crua"]])""",
    ),
    (
        "md",
        """Pela média crua, o melhor canal seria o YouTube · Games. Só que a mediana dele é menos de um terço da média, e nos outros dois canais do YouTube um único post responde pela maior parte da soma. A média crua, com tão poucos posts, mede o maior post de cada canal.

A tabela seguinte mostra a mediana por dia de publicação. O YouTube · Games mudou de patamar em um dia: é o tipo de coisa que nenhum modelo de média fixa enxerga, e fica registrado como limitação.""",
    ),
    (
        "code",
        """display(r["lote_por_dia"])""",
    ),
    (
        "md",
        """### Posts jovens: o número ainda vai subir

As duas coletas, com cerca de 13 horas entre elas, mostram quanto as views de um mesmo post cresceram conforme a idade que ele tinha na primeira.""",
    ),
    (
        "code",
        """display(r["crescimento"])
tabela = r["tabela"]
print("posts com 24 h ou mais (valor final):", int((~tabela["jovem"]).sum()))
print("posts de 12 a 24 h (entre o observado e 5x):", int(np.isfinite(tabela["teto"]).sum()))
print("posts com menos de 12 h (só piso):", int((tabela["jovem"] & ~np.isfinite(tabela["teto"])).sum()))""",
    ),
    (
        "md",
        """Quem já tinha 24 horas quase não muda. Daí o tratamento: `y = log(1 + views)`; post com 24 horas ou mais entra com o valor observado; post de 12 a 24 horas entra como "entre o observado e 5 vezes o observado" (o maior crescimento visto depois de 12 horas foi de 3,1 vezes; 5 é folga); post com menos de 12 horas entra só como piso. No amostrador, o valor final desses posts é sorteado a cada passo de uma normal truncada (aumento de dados). A seção 5 mostra o quanto a conclusão depende dessa escolha.""",
    ),
    (
        "md",
        """## 2. Três formas de estimar o desempenho de cada canal

- **Sem agrupamento:** cada canal sozinho, com a sua média e a sua dispersão.
- **Agrupamento total:** uma média só para os seis.
- **Hierárquico (agrupamento parcial):** cada canal tem a sua média, mas as médias dos canais de uma plataforma vêm de uma distribuição comum, com desvio `tau`. Quanto menos dado o canal tem e quanto menor `tau`, mais a estimativa dele é puxada para a média da plataforma.

As três usam a mesma verossimilhança e o mesmo amostrador; muda só o que é compartilhado. O "post típico" é `exp(theta) - 1`: a mediana das views, não a média.""",
    ),
    (
        "code",
        """est = r["estimativas"]
formas = ["sem_agrupamento", "hierarquico", "agrupamento_total"]
quadro = pd.concat(
    {analise.NOME_DA_FORMA[f]: est[est["forma"] == f].set_index("canal")[["tipico", "tipico_inferior", "tipico_superior"]] for f in formas}, axis=1
).reindex(dados.CANAIS)
display(quadro.round(0))
display(r["encolhimento"])
figuras["estimativas"]""",
    ),
    (
        "md",
        """Sozinho, o YouTube · Games parece valer várias vezes o YouTube · Ciência, com intervalos enormes. O modelo hierárquico puxa os três canais do YouTube para perto um do outro (cerca de 80% do caminho até a média da plataforma) e estreita os intervalos. No TikTok os três canais já estavam juntos. O agrupamento total fica no meio do nada: não descreve nenhuma das duas plataformas.

## 3. O amostrador convergiu?

Quatro cadeias partindo de pontos espalhados. R-hat perto de 1 diz que elas contam a mesma história; o tamanho efetivo diz a quantas amostras independentes as cadeias equivalem. `tau` é o parâmetro mais difícil (é o que usa Metropolis), e por isso tem o menor tamanho efetivo.""",
    ),
    (
        "code",
        """display(r["diagnosticos"]["hierarquico"])
print("aceitação do Metropolis (tau):", round(float(r["principal"].posteriori.aceitacao_tau.mean()), 3))
display(pd.DataFrame(r["pior_diagnostico"]).T)""",
    ),
    (
        "md",
        """## 4. Sensibilidade à priori de tau

Com seis canais (três por plataforma), há pouquíssima informação sobre o quanto os canais diferem entre si. O mesmo modelo, com três prioris razoáveis para `tau`:""",
    ),
    (
        "code",
        """display(r["prioris"].T)
figuras["prioris_de_tau"]""",
    ),
    (
        "md",
        """A mediana de `tau` muda pouco, mas o limite superior quase dobra da semi-normal para a uniforme: os dados dizem que `tau` provavelmente é pequeno e não conseguem descartar que seja grande. A decisão (quem lidera e com que probabilidade) quase não muda entre as prioris.

## 5. Sensibilidade ao tratamento dos posts jovens

Aqui a conclusão muda mais.""",
    ),
    (
        "code",
        """display(r["jovens"].T)""",
    ),
    (
        "md",
        """A plataforma vence em qualquer tratamento. O canal líder dentro do TikTok, não: troca conforme os posts do último dia entram ou não. E o nível do YouTube varia por um fator de quatro a cinco. É a amostra pequena aparecendo: um dia de publicação muda a ordem.

## 6. Checagem preditiva a posteriori

O modelo ajustado gera dados parecidos com os reais? Simulam-se milhares de conjuntos de dados a partir da posteriori e compara-se a dispersão e a cauda com o observado nos posts maduros.""",
    ),
    (
        "code",
        """display(r["checagem"])
print("Só com os posts maduros no ajuste:")
display(r["checagem_so_maduros"])
figuras["checagem_preditiva"]""",
    ),
    (
        "md",
        """A dispersão do YouTube e os extremos dele passam. Três checagens falham: no TikTok o modelo espera mais dispersão e posts maiores do que os posts maduros mostram (um post jovem que ficou em 6 views infla o desvio, e os posts maduros batem num teto perto de 650 views), e no YouTube 60% dos posts ficam com até 5 views, mais do que uma normal em log produz. O modelo serve para comparar o centro dos canais; não serve para descrever a forma da distribuição.

## 7. Validação: qual forma prevê melhor um post que ficou de fora?

Duas validações. Na primeira, um corte sai por vez (com os posts dele nas duas plataformas) e os posts maduros desse corte são previstos. Na segunda, o modelo treina com a primeira metade dos posts maduros, por data, e prevê a segunda metade. O erro é o erro absoluto em log; a diferença para o hierárquico traz intervalo de 95% por bootstrap de cortes.""",
    ),
    (
        "code",
        """colunas = ["erro_absoluto_medio", "fator_de_erro", "diferenca_de_erro_para_o_hierarquico", "diferenca_de_erro_inferior", "diferenca_de_erro_superior", "log_densidade_media"]
ordem = ["hierarquico", "por_plataforma", "media_crua", "sem_agrupamento", "hierarquico_sem_plataforma", "agrupamento_total"]
display(r["validacao_cruzada"].loc[ordem, colunas].rename(index=analise.NOME_DA_FORMA))
print(r["info_do_tempo"])
display(r["validacao_no_tempo"].loc[ordem, colunas].rename(index=analise.NOME_DA_FORMA))
figuras["validacao"]""",
    ),
    (
        "md",
        """O modelo hierárquico **empata** com as alternativas simples: uma média por plataforma erra o mesmo, e a média crua do canal também (o intervalo da diferença inclui o zero nas duas validações). Quem perde com clareza é juntar tudo e ignorar a plataforma. O que o modelo hierárquico entrega a mais não é previsão de ponto: é a incerteza de cada canal e a probabilidade de cada um ser o melhor.

## 8. A decisão""",
    ),
    (
        "code",
        """d = r["decisao"]
display(pd.DataFrame({"P(ser o melhor)": d["p_melhor"], "views perdidas por post, em média, apostando nele": d["perda_esperada_por_post"]}))
print("P(o melhor canal é do TikTok):", round(d["p_melhor_e_do_tiktok"], 4))
print("post típico do TikTok sobre o do YouTube:", {k: round(v, 2) for k, v in d["tiktok_sobre_youtube"].items()})
print("dentro do TikTok, melhor sobre pior:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d["dentro_da_plataforma"]["tiktok"].items()})
print("P(melhor nicho, somando as duas plataformas):", {k: round(v, 3) for k, v in d["p_melhor_nicho"].items()})""",
    ),
    (
        "md",
        """A plataforma está decidida. O canal dentro do TikTok, não: o líder não chega a 50%, e a diferença entre o melhor e o pior canal do TikTok é pequena.

Um alerta sobre o objetivo. Tudo acima é o post típico (mediana). Se o objetivo fosse a soma de views, a conta seria `exp(theta + sigma²/2)`, que no YouTube é quase toda cauda extrapolada:""",
    ),
    (
        "code",
        """display(pd.DataFrame(r["media_no_modelo"]).T.round(2))""",
    ),
    (
        "md",
        """Os intervalos do YouTube vão de cerca de cem a mais de dez mil views. Com esta amostra, a média de views do YouTube não é estimável; a decisão fica no post típico, e isso está declarado como limitação.

## 9. Quanto custa decidir cedo: simulação de alocação

Cada "mundo" é uma amostra da posteriori: um desempenho plausível para cada canal. Em cada mundo, três regras publicam os posts das próximas quatro semanas da agenda, uma de cada vez, começando do zero:

- **repartição igual** (a regra atual);
- **tudo no líder depois de N posts por canal** (a tentação);
- **amostragem de Thompson** (a cada post, sorteia um desempenho plausível de cada canal e publica no que saiu melhor).

O arrependimento é o que se deixou de ganhar, em views do post típico, por não publicar sempre no melhor canal daquele mundo.""",
    ),
    (
        "code",
        """print(r["horizonte"])
sim = r["simulacao"]
display(sim[["politica", "parcela_perdida_do_ideal", "arrependimento_medio", "arrependimento_p10", "arrependimento_mediano", "arrependimento_p90", "p_termina_em_canal_ruim", "p_termina_na_plataforma_errada", "p_perde_menos_que_thompson"]])
figuras["arrependimento"]""",
    ),
    (
        "code",
        """figuras["decidir_cedo"]""",
    ),
    (
        "md",
        """Três leituras:

1. Decidir cedo é uma aposta: na maioria dos mundos, apostar tudo no líder depois de 2 ou 3 posts por canal perde menos que o Thompson (veja a mediana e a última coluna). Mas em cerca de um a cada quatro ou cinco mundos a aposta termina num canal ruim, e aí a perda é várias vezes maior (veja o percentil 90). Na média, o Thompson ganha.
2. Esperar demais também custa: cada post gasto para "ter certeza" num canal fraco é perda garantida.
3. A repartição igual é a pior regra só se o esforço for transferível. Na operação real, o mesmo corte vai para as duas plataformas quase de graça; a escolha que custa é o nicho, e para o nicho os dados ainda não separam ninguém.

## 10. Onde este estudo falha

- 64 posts publicados em quatro dias: um único dia mudou o patamar do YouTube · Games, e o modelo supõe canais estáveis.
- O canal líder dentro do TikTok troca conforme o tratamento dos posts jovens.
- O modelo não reproduz a forma dos dados (três de seis checagens preditivas falham).
- O objetivo é o post típico; a soma de views, dominada pela cauda, não é estimável com esta amostra.
- `tau` é comum às duas plataformas e mal identificado com seis canais.
- A simulação sorteia os mundos da própria posteriori e herda todas as limitações acima.

O README traz a lista completa.""",
    ),
]


def main() -> None:
    caderno = new_notebook()
    caderno.cells = [new_markdown_cell(texto) if tipo == "md" else new_code_cell(texto) for tipo, texto in CELULAS]
    caderno.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    caderno.metadata["language_info"] = {"name": "python"}
    nbformat.write(caderno, RAIZ / "analise.ipynb")
    print(f"Montado: {RAIZ / 'analise.ipynb'} ({len(caderno.cells)} células)")


if __name__ == "__main__":
    main()
