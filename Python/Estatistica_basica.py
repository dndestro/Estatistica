import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def plot_qq_with_ci(data, dist_name, ax, alpha=0.05):
    """
    Gera um Q-Q Plot para uma distribuição teórica específica com intervalo de confiança.

    Parâmetros:
    - data: array/lista com os dados amostrais.
    - dist_name: nome da distribuição no scipy.stats ('norm', 'lognorm', 'expon', 'weibull_min').
    - ax: eixo do matplotlib onde o gráfico será plotado.
    - alpha: nível de significância para o intervalo de confiança (ex: 0.05 para 95% de IC).
    """
    n = len(data)
    sorted_data = np.sort(data)

    # Posições de plotagem (Posição de Blom)
    i = np.arange(1, n + 1)
    p = (i - 0.375) / (n + 0.25)

    # Ajuste dos parâmetros da distribuição com base nos dados
    dist = getattr(stats, dist_name)
    if dist_name == 'norm':
        loc, scale = dist.fit(data)
        args = (loc, scale)
    elif dist_name == 'lognorm':
        shape, loc, scale = dist.fit(data, floc=0)
        args = (shape, loc, scale)
    elif dist_name == 'expon':
        loc, scale = dist.fit(data, floc=0)
        args = (loc, scale)
    elif dist_name == 'weibull_min':
        c, loc, scale = dist.fit(data, floc=0)
        args = (c, loc, scale)
    else:
        args = dist.fit(data)

    # Quantis teóricos da distribuição
    theo_quantiles = dist.ppf(p, *args)

    # Intervalos de confiança usando a distribuição Beta para estatísticas de ordem
    p_lower = stats.beta.ppf(alpha / 2, i, n - i + 1)
    p_upper = stats.beta.ppf(1 - alpha / 2, i, n - i + 1)

    # Tratamento de limites para evitar estouro numérico
    p_lower = np.clip(p_lower, 1e-8, 1 - 1e-8)
    p_upper = np.clip(p_upper, 1e-8, 1 - 1e-8)

    theo_lower = dist.ppf(p_lower, *args)
    theo_upper = dist.ppf(p_upper, *args)

    # Gráfico de dispersão dos dados
    ax.scatter(theo_quantiles, sorted_data, color='#1f77b4',
               alpha=0.7, s=30, label='Dados Amostrais')

    # Linha de referência diagonal (45 graus / identidade)
    min_val = min(theo_quantiles.min(), sorted_data.min())
    max_val = max(theo_quantiles.max(), sorted_data.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', lw=2, label='Linha Teórica (1:1)')

    # Faixa do Intervalo de Confiança
    conf_pct = int((1 - alpha) * 100)
    ax.fill_between(theo_quantiles, theo_lower, theo_upper,
                    color='gray', alpha=0.3, label=f'IC {conf_pct}%')

    # Estilização do gráfico
    ax.set_title(f'Q-Q Plot: {dist_name.capitalize()}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Quantis Teóricos', fontsize=10)
    ax.set_ylabel('Quantis Amostrais', fontsize=10)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)


# ==========================================
# Exemplo de Uso
# ==========================================
if __name__ == '__main__':
    # Gerando dados de exemplo (ex: seguindo uma distribuição Weibull)
    np.random.seed(42)
    meus_dados = stats.weibull_min.rvs(c=2.5, scale=10, size=150)

    # AJUSTE DO INTERVALO DE CONFIANÇA AQUI:
    # alpha = 0.05 corresponde a 95% de confiança (IC de 95%)
    # Para 99% de confiança, altere para alpha = 0.01
    NIVEL_ALPHA = 0.05

    # Criando o painel com 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    distribuicoes = ['norm', 'lognorm', 'expon', 'weibull_min']

    for ax, d_name in zip(axes.flatten(), distribuicoes):
        plot_qq_with_ci(meus_dados, d_name, ax, alpha=NIVEL_ALPHA)

    plt.tight_layout()
    plt.show()
