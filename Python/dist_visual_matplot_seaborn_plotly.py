import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Tenta importar o Plotly; caso não esteja instalado, avisa o usuário
try:
    import plotly.graph_objects as go
    PLOTLY_DISPONIVEL = True
except ImportError:
    PLOTLY_DISPONIVEL = False

# Escolha o motor de visualização desejado: 'matplotlib', 'seaborn' ou 'plotly'
ENGINE = 'seaborn'  # Altere para 'matplotlib' ou 'plotly' conforme preferir


def plot_qq_with_ci(data, dist_name, ax=None, alpha=0.05, engine='matplotlib'):
    """
    Gera um Q-Q Plot para uma distribuição teórica específica com intervalo de confiança.

    Parâmetros:
    - data: array/lista com os dados amostrais.
    - dist_name: nome da distribuição no scipy.stats ('norm', 'lognorm', 'expon', 'weibull_min').
    - ax: eixo do matplotlib onde o gráfico será plotado (usado para matplotlib/seaborn).
    - alpha: nível de significância para o intervalo de confiança (ex: 0.05 para 95% de IC).
    - engine: 'matplotlib', 'seaborn' ou 'plotly'.
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

    conf_pct = int((1 - alpha) * 100)
    min_val = min(theo_quantiles.min(), sorted_data.min())
    max_val = max(theo_quantiles.max(), sorted_data.max())

    # Renderização via Plotly
    if engine == 'plotly':
        if not PLOTLY_DISPONIVEL:
            raise ImportError(
                "A biblioteca 'plotly' não está instalada. Execute 'pip install plotly' no terminal.")

        fig_plotly = go.Figure()

        # Intervalo de confiança (área preenchida)
        fig_plotly.add_trace(go.Scatter(
            x=np.concatenate([theo_quantiles, theo_quantiles[::-1]]),
            y=np.concatenate([theo_upper, theo_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(150, 150, 150, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name=f'IC {conf_pct}%'
        ))

        # Linha teórica 1:1
        fig_plotly.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Linha Teórica (1:1)'
        ))

        # Dados Amostrais
        fig_plotly.add_trace(go.Scatter(
            x=theo_quantiles, y=sorted_data,
            mode='markers',
            marker=dict(color='#1f77b4', size=8, opacity=0.7),
            name='Dados Amostrais'
        ))

        fig_plotly.update_layout(
            title=f'Q-Q Plot: {dist_name.capitalize()}',
            xaxis_title='Quantis Teóricos',
            yaxis_title='Quantis Amostrais',
            template='plotly_white'
        )
        fig_plotly.show()
        return

    # Renderização via Seaborn
    sns.set_theme()
    if engine == 'seaborn':
        sns.set_theme(style="whitegrid")
        sns.scatterplot(x=theo_quantiles, y=sorted_data, color='#1f77b4',
                        alpha=0.7, s=40, ax=ax, label='Dados Amostrais')
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', lw=2, label='Linha Teórica (1:1)')
        ax.fill_between(theo_quantiles, theo_lower, theo_upper,
                        color='gray', alpha=0.3, label=f'IC {conf_pct}%')

        ax.set_title(f'Q-Q Plot: {dist_name.capitalize()}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Quantis Teóricos', fontsize=10)
        ax.set_ylabel('Quantis Amostrais', fontsize=10)
        ax.legend(loc='upper left', frameon=True)
        return

    # Renderização padrão via Matplotlib puro
    ax.scatter(theo_quantiles, sorted_data, color='#1f77b4',
               alpha=0.7, s=30, label='Dados Amostrais')

    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', lw=2, label='Linha Teórica (1:1)')

    ax.fill_between(theo_quantiles, theo_lower, theo_upper,
                    color='gray', alpha=0.3, label=f'IC {conf_pct}%')

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
    NIVEL_ALPHA = 0.05

    distribuicoes = ['norm', 'lognorm', 'expon', 'weibull_min']
    ENGINE = 'seaborn'
    # Se escolher Plotly, ele gera gráficos interativos separados para cada distribuição
    if ENGINE == 'plotly':
        for d_name in distribuicoes:
            plot_qq_with_ci(meus_dados, d_name,
                            alpha=NIVEL_ALPHA, engine='plotly')
    else:
        # Criando o painel com 4 subplots para Matplotlib ou Seaborn
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for ax, d_name in zip(axes.flatten(), distribuicoes):
            plot_qq_with_ci(meus_dados, d_name, ax=ax,
                            alpha=NIVEL_ALPHA, engine=ENGINE)

        plt.tight_layout()
        plt.show()
