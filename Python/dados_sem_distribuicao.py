import numpy as np
import scipy.integrate as integrate
from scipy.stats import gaussian_kde, genpareto
from statsmodels.distributions.empirical_distribution import ECDF
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------------------------------
# 1. Preparação dos Dados (Mistura não-padrão)
# -------------------------------------------------------------------------
np.random.seed(42)
dados = np.random.exponential(
    scale=2.0, size=1000) + np.random.normal(0, 0.5, 1000)
dados = np.abs(dados)

valor_alvo = 3.0
print(f"Tamanho da amostra: {len(dados)} observações")
print(f"Valor-alvo sob análise: {valor_alvo}\n")

# -------------------------------------------------------------------------
# 1. Função de Distribuição Acumulada Empírica (ECDF)
# -------------------------------------------------------------------------
ecdf = ECDF(dados)
prob_ecdf = ecdf(valor_alvo)
print(
    f"1. ECDF -> P(X <= {valor_alvo}): {prob_ecdf:.4f} ({prob_ecdf*100:.2f}%)")

# -------------------------------------------------------------------------
# 2. Estimativa de Densidade por Kernel (KDE)
# -------------------------------------------------------------------------
kde = gaussian_kde(dados)
prob_kde, _ = integrate.quad(kde.evaluate, dados.min(), valor_alvo)
print(f"2. KDE  -> P(X <= {valor_alvo}): {prob_kde:.4f} ({prob_kde*100:.2f}%)")

# -------------------------------------------------------------------------
# 3. Amostragem Bootstrap (Cálculo de Intervalo de Confiança)
# -------------------------------------------------------------------------
n_bootstrap = 1000
bootstrap_probs = [ECDF(np.random.choice(dados, size=len(dados), replace=True))(
    valor_alvo) for _ in range(n_bootstrap)]
ci_inferior = np.percentile(bootstrap_probs, 2.5)
ci_superior = np.percentile(bootstrap_probs, 97.5)
print(f"3. Bootstrap (IC 95%) para P(X <= {valor_alvo}):")
print(
    f"   Limite Inferior: {ci_inferior:.4f} | Limite Superior: {ci_superior:.4f}")

# -------------------------------------------------------------------------
# 4. Teoria dos Valores Extremos (EVT / GPD para Caudas)
# -------------------------------------------------------------------------
# Definindo um limiar alto (ex: percentil 90 para focar na cauda)
limiar = np.percentile(dados, 90)
excedencias = dados[dados > limiar] - limiar

if len(excedencias) > 10:
    c, loc, scale = genpareto.fit(excedencias)

    # Queremos estimar a probabilidade de um evento extremo na cauda (ex: percentil 98)
    valor_extremo = np.percentile(dados, 98)

    # Probabilidade condicional de exceder o valor extremo dado o limiar
    prob_condicional = 1 - \
        genpareto.cdf(valor_extremo - limiar, c, loc=loc, scale=scale)
    # Probabilidade total considerando a proporção de dados que ultrapassam o limiar (10%)
    prob_evt = (1 - 0.90) * prob_condicional

    print(f"\n4. EVT (Cauda / Eventos Raros):")
    print(f"   Limiar de corte (P90): {limiar:.4f}")
    print(
        f"   P(X > {valor_extremo:.4f}) no extremo: {prob_evt:.6f} ({prob_evt*100:.4f}%)")
else:
    print("\n4. EVT: Amostra insuficiente na cauda para ajuste estável da GPD.")


# -------------------------------------------------------------------------
# 3. Construção do Painel Gráfico com Plotly (2x2)
# -------------------------------------------------------------------------
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "1. Função de Distribuição Acumulada Empírica (ECDF)",
        "2. Densidade por Kernel (KDE)",
        "3. Amostragem Bootstrap (IC 95% para Probabilidade)",
        "4. Teoria dos Valores Extremos (EVT / GPD na Cauda)"
    )
)

# --- Subplot 1: ECDF ---
x_sorted = np.sort(dados)
y_ecdf = np.arange(1, len(dados) + 1) / len(dados)
fig.add_trace(go.Scatter(x=x_sorted, y=y_ecdf, mode='lines',
              name='ECDF', line=dict(color='royalblue', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=[valor_alvo], y=[prob_ecdf], mode='markers', marker=dict(
    color='red', size=10), name=f'Alvo ({prob_ecdf:.2f})'), row=1, col=1)

# --- Subplot 2: KDE ---
x_kde = np.linspace(dados.min(), dados.max(), 300)
y_kde = kde.evaluate(x_kde)
fig.add_trace(go.Scatter(x=x_kde, y=y_kde, mode='lines', name='KDE PDF',
              line=dict(color='darkorange', width=2)), row=1, col=2)
# Preencher área até o valor alvo
mask = x_kde <= valor_alvo
fig.add_trace(go.Scatter(x=np.concatenate([[x_kde[0]], x_kde[mask], [valor_alvo]]),
                         y=np.concatenate([[0], y_kde[mask], [0]]),
                         fill='tozeroy', mode='none', fillcolor='rgba(255, 165, 0, 0.3)', name=f'P(X<=3)={prob_kde:.2f}'), row=1, col=2)

# --- Subplot 3: Bootstrap ---
fig.add_trace(go.Histogram(x=bootstrap_probs, marker_color='forestgreen',
              opacity=0.75, name='Bootstrap Dist.'), row=2, col=1)
fig.add_vline(x=ci_inferior, line_dash="dash", line_color="red",
              annotation_text=f"IC 2.5%: {ci_inferior:.2f}", row=2, col=1)
fig.add_vline(x=ci_superior, line_dash="dash", line_color="red",
              annotation_text=f"IC 97.5%: {ci_superior:.2f}", row=2, col=1)

# --- Subplot 4: EVT / GPD ---
x_gpd = np.linspace(0, max(excedencias), 100)
y_gpd_pdf = genpareto.pdf(x_gpd, c, loc=loc, scale=scale)
fig.add_trace(go.Histogram(x=excedencias, histnorm='probability density',
              marker_color='purple', opacity=0.6, name='Excedências (Acima P90)'), row=2, col=2)
fig.add_trace(go.Scatter(x=x_gpd, y=y_gpd_pdf, mode='lines',
              name='GPD Fit', line=dict(color='crimson', width=2)), row=2, col=2)

# Configurações gerais do layout
fig.update_layout(
    title_text="<b>Análise Não-Paramétrica de Dados e Teoria de Valores Extremos (EVT)</b>",
    template="plotly_white",
    height=800,
    showlegend=False
)

# Exibir gráfico (se executado em ambiente interativo Jupyter/Streamlit)
fig.show()
