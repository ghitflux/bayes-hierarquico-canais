import numpy as np
import pandas as pd
import pytest

from bayes_canais import analise, dados
from bayes_canais.dados import CANAIS, NICHOS, PLATAFORMAS


def _posts(gerador: np.random.Generator, por_canal: int = 8) -> pd.DataFrame:
    """Posts sintéticos em memória: TikTok perto de 500 views, YouTube perto de 20 e muito disperso."""
    linhas = []
    for p, plataforma in enumerate(PLATAFORMAS):
        for nicho in NICHOS:
            for i in range(por_canal):
                # Os dois últimos posts de cada canal são jovens: um com 18 h, outro com 3 h.
                horas = 18.0 if i == por_canal - 2 else 3.0 if i == por_canal - 1 else 30.0 + 6 * i
                log_views = gerador.normal(6.2, 0.4) if p == 0 else gerador.normal(3.0, 2.0)
                linhas.append(
                    {
                        "coleta": "2026-01-02",
                        "corte": f"{nicho}-{i}",
                        "nicho": nicho,
                        "plataforma": plataforma,
                        "publicado_em": f"2026-01-01T{i:02d}:00:00",
                        "horas_no_ar": horas,
                        "views": int(np.expm1(max(log_views, 0.0))),
                    }
                )
    return pd.DataFrame(linhas)


@pytest.fixture(scope="module")
def tabela() -> pd.DataFrame:
    return dados.preparar(_posts(np.random.default_rng(0)))


CURTO = analise.Orcamento(cadeias=4, iteracoes=600, aquecimento=300)


def test_preparar_marca_jovens_e_poe_teto_so_em_quem_tem_entre_12_e_24_horas(tabela):
    assert len(tabela) == 48 and list(tabela["canal"].unique()) == CANAIS
    assert tabela["y"].to_numpy() == pytest.approx(np.log1p(tabela["views"].to_numpy()))
    assert tabela["jovem"].sum() == 12
    com_teto = np.isfinite(tabela["teto"])
    assert (tabela.loc[com_teto, "horas_no_ar"] == 18.0).all() and com_teto.sum() == 6
    assert (tabela.loc[com_teto, "teto"] - tabela.loc[com_teto, "y"]).to_numpy() == pytest.approx(np.log(dados.FATOR_MAXIMO))

    sem_teto = dados.preparar(_posts(np.random.default_rng(0)), fator_maximo=None)
    assert not np.isfinite(sem_teto["teto"]).any()


def test_crescimento_entre_coletas_separa_por_idade():
    antes = pd.DataFrame(
        {
            "coleta": "2026-01-01",
            "corte": ["a", "b", "c", "d"],
            "plataforma": "tiktok",
            "horas_no_ar": [2.0, 15.0, 30.0, 40.0],
            "views": [9, 99, 199, 499],
        }
    )
    depois = antes.assign(coleta="2026-01-02", horas_no_ar=antes["horas_no_ar"] + 13, views=[99, 199, 199, 549])
    resumo = dados.crescimento_entre_coletas(pd.concat([antes, depois]))
    assert resumo.loc["menos de 12 h", "fator_maximo"] == pytest.approx(10.0)
    assert resumo.loc["12 a 24 h", "fator_maximo"] == pytest.approx(2.0)
    assert resumo.loc["24 h ou mais", "posts"] == 2
    assert resumo.loc["24 h ou mais", "fator_maximo"] == pytest.approx(1.1)
    assert resumo.attrs["horas_entre_coletas"] == pytest.approx(13.0)
    assert dados.crescimento_entre_coletas(antes) is None  # uma coleta só


def test_montar_define_quem_estima_cada_canal_em_cada_forma(tabela):
    esperado = {
        "sem_agrupamento": (6, [0, 1, 2, 3, 4, 5], False),
        "agrupamento_total": (1, [0, 0, 0, 0, 0, 0], False),
        "por_plataforma": (2, [0, 0, 0, 1, 1, 1], False),
        "hierarquico": (6, [0, 1, 2, 3, 4, 5], True),
        "hierarquico_sem_plataforma": (6, [0, 1, 2, 3, 4, 5], True),
    }
    for forma, (grupos, mapa, estima_tau) in esperado.items():
        estrutura, grupo_do_canal, tau_fixo = analise.montar(tabela, forma)
        assert estrutura.grupos == grupos
        assert list(grupo_do_canal) == mapa
        assert (tau_fixo is None) == estima_tau
        assert estrutura.censurado.sum() == 12
    assert analise.montar(tabela, "hierarquico")[0].blocos == 2
    assert analise.montar(tabela, "hierarquico_sem_plataforma")[0].blocos == 1
    with pytest.raises(ValueError):
        analise.montar(tabela, "outra")


def test_probabilidade_de_ser_o_melhor_conta_as_amostras_em_que_cada_canal_vence():
    theta = np.array([[3.0, 1.0, 2.0], [3.0, 4.0, 2.0], [5.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    assert analise.probabilidade_de_ser_o_melhor(theta) == pytest.approx([0.5, 0.25, 0.25])


def test_modelo_hierarquico_separa_as_plataformas_e_encolhe_os_canais(tabela):
    ajustes = {forma: analise.ajustar(tabela, forma, orcamento=CURTO, semente=1) for forma in ("sem_agrupamento", "hierarquico", "agrupamento_total")}
    decisao = analise.decisao(ajustes["hierarquico"])
    assert decisao["p_melhor_e_do_tiktok"] > 0.99
    assert sum(decisao["p_melhor"].values()) == pytest.approx(1.0)
    assert sum(decisao["p_melhor_nicho"].values()) == pytest.approx(1.0)
    assert decisao["tiktok_sobre_youtube"]["fator"] > 5

    # Encolher aproxima os canais do YouTube entre si e estreita os intervalos.
    estimativas = analise.estimativas_por_canal(ajustes).pivot(index="canal", columns="forma", values="theta").reindex(CANAIS)
    youtube = estimativas.iloc[3:]
    assert np.ptp(youtube["hierarquico"]) < np.ptp(youtube["sem_agrupamento"])
    assert estimativas["agrupamento_total"].nunique() == 1
    larguras = analise.encolhimento(ajustes)
    assert (larguras["largura_do_intervalo_hierarquico"].iloc[3:] < larguras["largura_do_intervalo_sem_agrupamento"].iloc[3:]).all()

    diagnosticos = analise.diagnosticos_do_ajuste(ajustes["hierarquico"])
    assert "tau" in diagnosticos.index and len(diagnosticos) == 11
    assert analise.pior_diagnostico(diagnosticos)["maior_rhat"] < 1.1


def test_checagem_preditiva_aprova_dados_gerados_pelo_proprio_modelo(tabela):
    ajuste = analise.ajustar(tabela, "hierarquico", orcamento=CURTO, semente=2)
    checagem, simulado = analise.checagem_preditiva(ajuste, tabela, replicas=1000, semente=3)
    assert len(checagem) == 6 and set(simulado) == set(checagem.index)
    dispersao = checagem.loc[["TikTok: desvio do log das views", "YouTube: desvio do log das views"], "p_preditivo"]
    assert ((dispersao > 0.02) & (dispersao < 0.98)).all()


def test_validacao_preve_so_posts_maduros_e_nao_usa_o_corte_previsto(tabela):
    pequena = tabela[tabela["corte"].isin([f"{n}-{i}" for n in NICHOS for i in (0, 1, 6, 7)])].reset_index(drop=True)
    orcamento = analise.Orcamento(cadeias=2, iteracoes=150, aquecimento=100)
    previsoes = analise.validacao_cruzada(pequena, ["media_crua", "hierarquico"], orcamento=orcamento)
    maduros = pequena[~pequena["jovem"]]
    assert len(previsoes) == 2 * len(maduros)
    assert set(previsoes["corte"]) == set(maduros["corte"])

    # A média crua de um canal, sem o corte previsto, é reproduzível à mão.
    linha = previsoes[(previsoes["forma"] == "media_crua")].iloc[0]
    resto = pequena[(pequena["canal"] == linha["canal"]) & (pequena["corte"] != linha["corte"])]
    assert linha["previsto"] == pytest.approx(resto["y"].mean())

    resumo = analise.resumo_da_validacao(previsoes, repeticoes=200)
    assert resumo.loc["hierarquico", "diferenca_de_erro_para_o_hierarquico"] == 0
    assert resumo.loc["hierarquico", "posts_previstos"] == len(maduros)


def test_validacao_no_tempo_treina_so_com_o_passado(tabela):
    orcamento = analise.Orcamento(cadeias=2, iteracoes=150, aquecimento=100)
    previsoes, info = analise.validacao_no_tempo(tabela, ["media_crua", "por_plataforma"], orcamento=orcamento)
    maduros = tabela[~tabela["jovem"]]
    assert info["posts_de_treino"] + info["posts_de_teste"] == len(maduros)
    assert info["posts_de_treino"] >= info["posts_de_teste"] > 0
    previstos = maduros[maduros["corte"].isin(previsoes["corte"])]
    assert (previstos["publicado_em"] > info["treina_ate"]).all()


def test_resumo_da_validacao_aponta_a_forma_que_erra_mais():
    gerador = np.random.default_rng(4)
    linhas = []
    for i in range(30):
        for canal in ("A", "B"):
            y = gerador.normal()
            for forma, ruido in (("hierarquico", 0.2), ("pior", 1.5)):
                previsto = y + gerador.normal(0, ruido)
                linhas.append({"corte": f"c{i}", "canal": canal, "forma": forma, "y": y, "previsto": previsto, "erro_absoluto": abs(y - previsto), "log_densidade": -abs(y - previsto)})
    resumo = analise.resumo_da_validacao(pd.DataFrame(linhas), repeticoes=500)
    assert resumo.loc["pior", "diferenca_de_erro_inferior"] > 0  # o intervalo da diferença exclui o zero
    assert resumo.loc["pior", "diferenca_de_log_densidade_superior"] < 0
    assert resumo.loc["hierarquico", "fator_de_erro"] == pytest.approx(np.exp(resumo.loc["hierarquico", "erro_absoluto_medio"]))


def test_horizonte_da_agenda_soma_as_semanas_seguintes_a_coleta():
    agenda = pd.DataFrame({"semana": ["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26"], "tiktok": [10, 20, 21, 30], "youtube": [10, 20, 20, 30]})
    horizonte = analise.horizonte_da_agenda(agenda, "2026-01-08", semanas=2)
    assert horizonte["posts_na_agenda"] == 81 and horizonte["horizonte"] == 84 and horizonte["de"] == "2026-01-12"


def test_simulacao_de_alocacao_compara_as_politicas_nos_mesmos_mundos(tabela):
    ajuste = analise.ajustar(tabela, "hierarquico", orcamento=CURTO, semente=5)
    resumo, curvas = analise.simulacao_de_alocacao(ajuste, horizonte=60, mundos=300, posts_iniciais=(2, 20), semente=6)
    assert list(resumo.index) == ["igual", "thompson", "lider_2"]  # 20 posts por canal não cabem em 60 posts
    assert curvas["thompson"].shape == (300, 60)
    assert resumo.loc["igual", "parcela_dos_posts_no_melhor_canal"] == pytest.approx(1 / 6)
    assert resumo.loc["thompson", "arrependimento_medio"] < resumo.loc["igual", "arrependimento_medio"]
    assert np.isnan(resumo.loc["igual", "p_termina_em_canal_ruim"])
    assert 0 <= resumo.loc["lider_2", "p_termina_em_canal_ruim"] <= resumo.loc["lider_2", "p_termina_fora_do_melhor_canal"]
