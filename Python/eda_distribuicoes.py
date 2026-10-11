#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise exploratória de dados (EDA) + identificação da distribuição estatística
que melhor se ajusta aos dados (testes estatísticos + análise gráfica).

Uso:
    python eda_distribuicoes.py                                  # dados simulados (demo)
    python eda_distribuicoes.py -a dados.csv -c minha_coluna
    python eda_distribuicoes.py -a dados.csv -c valor -g categoria --bootstrap 200

Dependências:
    pip install numpy pandas scipy matplotlib seaborn statsmodels
"""
import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# --------------------------------------------------------------------------- #
# Distribuições candidatas (nomes de scipy.stats)
# --------------------------------------------------------------------------- #
CANDIDATAS = [
    "norm", "lognorm", "gamma", "weibull_min", "expon", "logistic",
    "gumbel_r", "laplace", "t", "skewnorm", "invgauss", "rayleigh", "uniform",
]
# Para dados estritamente positivos, fixar loc=0 estabiliza o ajuste (MLE)
LOC_ZERO = {"lognorm", "gamma", "weibull_min", "invgauss"}


# --------------------------------------------------------------------------- #
# 1. Carga de dados
# --------------------------------------------------------------------------- #
def carregar_dados(arquivo, coluna, sep):
    """Lê o CSV (ou gera dados simulados se nenhum arquivo for informado)."""
    if arquivo is None:
        rng = np.random.default_rng(42)
        valor = np.concatenate(
            [rng.weibull(2.5, 400), rng.weibull(2.5, 400) * 3])
        df = pd.DataFrame({
            "valor": valor,
            "grupo": rng.choice(["A", "B"], valor.size),
            "outra": rng.normal(50, 10, valor.size),
        })
        print(">> Nenhum arquivo informado: usando dados simulados (coluna 'valor').\n")
        return df, "valor"

    df = pd.read_csv(arquivo, sep=sep)
    if coluna is None:
        numericas = df.select_dtypes("number").columns
        if len(numericas) == 0:
            raise SystemExit("Nenhuma coluna numérica encontrada.")
        coluna = numericas[0]
        print(
            f">> Coluna não informada: usando a primeira numérica ('{coluna}').\n")
    if coluna not in df.columns:
        raise SystemExit(
            f"Coluna '{coluna}' não existe. Colunas: {list(df.columns)}")
    return df, coluna


# --------------------------------------------------------------------------- #
# 2. EDA numérica
# --------------------------------------------------------------------------- #
def eda_basica(df, coluna, alpha=0.05):
    print("=" * 80, "\nVISÃO GERAL\n" + "=" * 80)
    print(f"Dimensão: {df.shape[0]} linhas x {df.shape[1]} colunas\n")
    print(df.dtypes.to_frame("tipo").T, "\n")
    faltantes = df.isna().sum()
    print("Valores ausentes:\n", faltantes[faltantes > 0].to_string()
          if faltantes.any() else "nenhum", "\n")

    x = df[coluna].dropna().astype(float)
    print("=" * 80, f"\nVARIÁVEL ANALISADA: {coluna}\n" + "=" * 80)
    print(x.describe(
        percentiles=[.01, .05, .25, .5, .75, .95, .99]).to_frame().T, "\n")
    print(f"Assimetria (skewness): {stats.skew(x):.4f}")
    print(f"Curtose (excesso):     {stats.kurtosis(x):.4f}")
    print(
        f"Coef. de variação:     {x.std() / x.mean():.4f}" if x.mean() != 0 else "")

    # Outliers pelo critério de Tukey (boxplot)
    q1, q3 = x.quantile([.25, .75])
    iqr = q3 - q1
    li, ls = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = x[(x < li) | (x > ls)]
    print(f"\nOutliers (Tukey 1.5*IQR): {len(out)} ({len(out) / len(x):.1%}) "
          f"fora de [{li:.3f}, {ls:.3f}]")

    if df.select_dtypes("number").shape[1] > 1:
        print("\nCorrelação (Pearson) entre variáveis numéricas:")
        print(df.select_dtypes("number").corr().round(3))
    return x


def testes_normalidade(x, alpha=0.05):
    """Bateria de testes de normalidade (H0: os dados vêm de uma Normal)."""
    print("\n" + "=" * 80, "\nTESTES DE NORMALIDADE (H0: dados ~ Normal)\n" + "=" * 80)
    linhas = []
    amostra = x if len(x) <= 5000 else x.sample(5000, random_state=1)
    w, p = stats.shapiro(amostra)
    linhas.append(("Shapiro-Wilk", w, p))
    k2, p = stats.normaltest(x)
    linhas.append(("D'Agostino-Pearson K²", k2, p))
    jb, p = stats.jarque_bera(x)
    linhas.append(("Jarque-Bera", jb, p))
    try:
        from statsmodels.stats.diagnostic import lilliefors
        d, p = lilliefors(x, dist="norm")
        linhas.append(("Lilliefors (KS c/ parâmetros estimados)", d, p))
    except ImportError:
        pass
    tab = pd.DataFrame(linhas, columns=["teste", "estatística", "p-valor"])
    tab["decisão"] = np.where(tab["p-valor"] < alpha,
                              "rejeita H0", "não rejeita H0")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    try:  # SciPy >= 1.17: p-valor interpolado
        ad = stats.anderson(x, dist="norm", method="interpolate")
        print(f"\nAnderson-Darling: A²={ad.statistic:.4f}, p-valor={ad.pvalue:.4g} -> "
              f"{'rejeita' if ad.pvalue < alpha else 'não rejeita'} H0")
    except TypeError:  # SciPy mais antigo: valores críticos tabelados
        ad = stats.anderson(x, dist="norm")
        crit = dict(zip(ad.significance_level, ad.critical_values))[5.0]
        print(f"\nAnderson-Darling: A²={ad.statistic:.4f} (valor crítico a 5% = {crit:.4f}) -> "
              f"{'rejeita' if ad.statistic > crit else 'não rejeita'} H0")


# --------------------------------------------------------------------------- #
# 3. Gráficos de EDA (inclui boxplot)
# --------------------------------------------------------------------------- #
def graficos_eda(df, coluna, grupo, saida, mostrar):
    x = df[coluna].dropna()
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"EDA — {coluna}", fontsize=14)

    sns.histplot(x, kde=True, stat="density", ax=ax[0, 0], color="steelblue")
    ax[0, 0].set_title("Histograma + KDE")

    sns.boxplot(x=x, ax=ax[0, 1], color="lightsteelblue", fliersize=4)
    sns.stripplot(x=x, ax=ax[0, 1], size=2, alpha=.25, color="k")
    ax[0, 1].set_title("Boxplot (com pontos individuais)")

    sns.violinplot(x=x, ax=ax[1, 0], inner="quartile", color="lightcoral")
    ax[1, 0].set_title("Violin plot")

    xs = np.sort(x.values)
    ax[1, 1].step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post")
    ax[1, 1].set_title("Distribuição acumulada empírica (ECDF)")
    fig.tight_layout()
    fig.savefig(saida / "01_eda_variavel.png", dpi=150)

    if grupo and grupo in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x=grupo, y=coluna, ax=ax, hue=grupo, legend=False)
        sns.stripplot(data=df, x=grupo, y=coluna, ax=ax,
                      size=2, alpha=.3, color="k")
        ax.set_title(f"Boxplot de {coluna} por {grupo}")
        fig.tight_layout()
        fig.savefig(saida / "02_boxplot_por_grupo.png", dpi=150)

    num = df.select_dtypes("number")
    if num.shape[1] > 1:
        n = num.shape[1]
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.5))
        for a, c in zip(np.atleast_1d(axes), num.columns):
            sns.boxplot(y=num[c], ax=a, color="lightsteelblue")
            a.set_title(c)
        fig.suptitle(
            "Boxplots das variáveis numéricas (escalas independentes)")
        fig.tight_layout()
        fig.savefig(saida / "03_boxplots_numericas.png", dpi=150)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(num.corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title("Matriz de correlação")
        fig.tight_layout()
        fig.savefig(saida / "04_correlacao.png", dpi=150)

    if not mostrar:
        plt.close("all")


# --------------------------------------------------------------------------- #
# 4. Ajuste de distribuições + testes de aderência
# --------------------------------------------------------------------------- #
def _ks_bootstrap(dist, params, x, kw, n_boot, seed=0):
    """
    p-valor do KS por bootstrap paramétrico. Necessário porque o p-valor
    "clássico" do KS é inválido (otimista) quando os parâmetros são estimados
    a partir dos mesmos dados.
    """
    rng = np.random.default_rng(seed)
    obs = stats.kstest(x, dist.cdf, args=params).statistic
    cont = validos = 0
    for _ in range(n_boot):
        amostra = dist.rvs(*params, size=len(x), random_state=rng)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = dist.fit(amostra, **kw)
            d = stats.kstest(amostra, dist.cdf, args=p).statistic
        except Exception:
            continue
        validos += 1
        cont += d >= obs
    return (cont + 1) / (validos + 1)


def ajustar_distribuicoes(x, candidatas, loc_zero=True, n_boot=0, alpha=0.05):
    x = np.asarray(x, dtype=float)
    n = len(x)
    linhas, parametros, kws = [], {}, {}

    for nome in candidatas:
        dist = getattr(stats, nome)
        kw = {"floc": 0} if (loc_zero and x.min() >
                             0 and nome in LOC_ZERO) else {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = dist.fit(x, **kw)
                ll = np.sum(dist.logpdf(x, *p))
        except Exception as e:
            print(f"  [!] {nome}: falha no ajuste ({e})")
            continue
        if not np.isfinite(ll):
            continue  # dados fora do suporte da distribuição

        k = len(p) - len(kw)  # nº de parâmetros livres
        ks = stats.kstest(x, nome, args=p)
        cvm = stats.cramervonmises(x, nome, args=p)
        linhas.append({
            "distribuicao": nome, "k": k, "loglik": ll,
            "AIC": 2 * k - 2 * ll, "BIC": k * np.log(n) - 2 * ll,
            "KS": ks.statistic, "KS_p": ks.pvalue,
            "CvM": cvm.statistic, "CvM_p": cvm.pvalue,
        })
        parametros[nome], kws[nome] = p, kw

    tab = pd.DataFrame(linhas).sort_values("AIC").reset_index(drop=True)
    tab["dAIC"] = tab["AIC"] - tab["AIC"].min()
    w = np.exp(-0.5 * tab["dAIC"])
    tab["peso_Akaike"] = w / w.sum()

    tab["KS_p_boot"] = np.nan
    if n_boot > 0:
        print(
            f"\nBootstrap paramétrico ({n_boot} réplicas) para as 3 melhores por AIC...")
        for i in range(min(3, len(tab))):
            nome = tab.loc[i, "distribuicao"]
            tab.loc[i, "KS_p_boot"] = _ks_bootstrap(
                getattr(stats, nome), parametros[nome], x, kws[nome], n_boot)

    tab["aceita_KS"] = np.where(
        tab["KS_p_boot"].fillna(tab["KS_p"]) > alpha, "sim", "não")
    return tab, parametros


def graficos_ajuste(x, tab, parametros, top_n, saida, mostrar):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    pe = (np.arange(1, n + 1) - 0.5) / n          # probabilidades empíricas
    top = tab["distribuicao"].head(top_n).tolist()
    grade = np.linspace(x.min(), x.max(), 500)

    # (a) Histograma com PDFs ajustadas  +  (b) ECDF vs CDFs teóricas
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].hist(x, bins="auto", density=True,
               alpha=.4, color="gray", label="dados")
    ax[1].step(x, np.arange(1, n + 1) / n, where="post",
               color="k", lw=2, label="empírica")
    for nome in top:
        d, p = getattr(stats, nome), parametros[nome]
        ax[0].plot(grade, d.pdf(grade, *p), lw=2, label=nome)
        ax[1].plot(grade, d.cdf(grade, *p), lw=1.8, ls="--", label=nome)
    ax[0].set_title(f"Histograma + PDFs ajustadas (top {top_n} por AIC)")
    ax[1].set_title("ECDF vs CDFs teóricas")
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(saida / "05_ajuste_pdf_cdf.png", dpi=150)

    # (c) Q-Q e P-P plots para cada distribuição do top
    fig, axes = plt.subplots(top_n, 2, figsize=(
        10, 4.2 * top_n), squeeze=False)
    for i, nome in enumerate(top):
        d, p = getattr(stats, nome), parametros[nome]
        teorico = d.ppf(pe, *p)
        lim = [min(x.min(), teorico.min()), max(x.max(), teorico.max())]
        axes[i, 0].scatter(teorico, x, s=8, alpha=.6)
        axes[i, 0].plot(lim, lim, "r--")
        axes[i, 0].set(title=f"Q-Q plot — {nome}", xlabel="quantis teóricos",
                       ylabel="quantis amostrais")
        axes[i, 1].scatter(d.cdf(x, *p), pe, s=8, alpha=.6)
        axes[i, 1].plot([0, 1], [0, 1], "r--")
        axes[i, 1].set(title=f"P-P plot — {nome}", xlabel="prob. teórica",
                       ylabel="prob. empírica")
    fig.tight_layout()
    fig.savefig(saida / "06_qq_pp_plots.png", dpi=150)

    # (d) Q-Q normal clássico (referência)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    stats.probplot(x, dist="norm", plot=ax)
    ax.set_title("Q-Q plot Normal (referência)")
    fig.tight_layout()
    fig.savefig(saida / "07_qq_normal.png", dpi=150)

    if not mostrar:
        plt.close("all")


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-a", "--arquivo",
                    help="CSV de entrada (padrão: dados simulados)")
    ap.add_argument("-c", "--coluna", help="coluna numérica a analisar")
    ap.add_argument("-g", "--grupo",
                    help="coluna categórica para boxplot por grupo")
    ap.add_argument("--sep", default=",", help="separador do CSV (padrão ',')")
    ap.add_argument("-s", "--saida", default="saida_eda",
                    help="pasta para salvar gráficos")
    ap.add_argument("--top", type=int, default=3,
                    help="nº de distribuições nos gráficos")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="réplicas p/ p-valor KS por bootstrap paramétrico (0 = desliga)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="nível de significância")
    ap.add_argument("--mostrar", action="store_true",
                    help="exibir gráficos na tela")
    args = ap.parse_args()

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    df, coluna = carregar_dados(args.arquivo, args.coluna, args.sep)
    x = eda_basica(df, coluna, args.alpha)
    testes_normalidade(x, args.alpha)
    graficos_eda(df, coluna, args.grupo, saida, args.mostrar)

    print("\n" + "=" * 80,
          "\nAJUSTE DE DISTRIBUIÇÕES (MLE) E TESTES DE ADERÊNCIA\n" + "=" * 80)
    tab, parametros = ajustar_distribuicoes(x, CANDIDATAS, n_boot=args.bootstrap,
                                            alpha=args.alpha)
    cols = ["distribuicao", "k", "AIC", "BIC", "dAIC", "peso_Akaike",
            "KS", "KS_p", "KS_p_boot", "CvM", "CvM_p", "aceita_KS"]
    print(tab[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    tab.to_csv(saida / "ranking_distribuicoes.csv", index=False)

    graficos_ajuste(x, tab, parametros, min(
        args.top, len(tab)), saida, args.mostrar)

    melhor = tab.loc[0, "distribuicao"]
    print(f"\n>> Melhor por AIC: {melhor}  | parâmetros (scipy): "
          f"{np.round(parametros[melhor], 4).tolist()}")
    print(f">> Gráficos e tabela salvos em: {saida.resolve()}")
    print("""
NOTAS DE INTERPRETAÇÃO
 - AIC/BIC comparam distribuições entre si (menor = melhor); dAIC < 2 indica
   modelos praticamente equivalentes.
 - KS e Cramér-von Mises avaliam aderência absoluta (H0: dados ~ distribuição).
   Com parâmetros estimados dos próprios dados, o p-valor "clássico" é otimista:
   use --bootstrap para um p-valor mais confiável.
 - Em amostras grandes qualquer teste rejeita desvios minúsculos; confie também
   nos gráficos (Q-Q e P-P devem seguir a diagonal, sobretudo nas caudas).
""")
    if args.mostrar:
        plt.show()


if __name__ == "__main__":
    main()
