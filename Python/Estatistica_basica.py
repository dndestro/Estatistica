import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Cria dados para teste do script
dados = np.random.normal(size=1000)
dados2 = np.random.normal(size=500)

# Resumo estatístico (equivalente ao summary())


def summary(data):
    print(f"Min:    {np.min(data):.4f}")
    print(f"Q1:     {np.percentile(data, 25):.4f}")
    print(f"Median: {np.median(data):.4f}")
    print(f"Mean:   {np.mean(data):.4f}")
    print(f"Q3:     {np.percentile(data, 75):.4f}")
    print(f"Max:    {np.max(data):.4f}")


summary(dados)

# Histograma
plt.figure()
plt.hist(dados, bins=30, color="steelblue", edgecolor="white")
plt.title("Distribuição Normal")
plt.xlabel("Valores")
plt.ylabel("Frequência")
plt.show()

# QQ Plot com envelope (equivalente ao qqPlot do car)
fig, ax = plt.subplots()

(osm, osr), (slope, intercept, r) = stats.probplot(dados, dist="norm")

# Linha de referência
ax.plot(osm, slope * np.array(osm) + intercept,
        color="red", label="Linha teórica")

# Pontos
ax.scatter(osm, osr, color="steelblue", s=10, label="Dados")

# Envelope de confiança 95%
n = len(dados)
confidence = 0.95
z = stats.norm.ppf((1 + confidence) / 2)
se = (1 / stats.norm.pdf(osm)) * \
    np.sqrt(
        np.array([p * (1 - p) / n for p in np.linspace(0.01, 0.99, len(osm))]))
upper = slope * np.array(osm) + intercept + z * se
lower = slope * np.array(osm) + intercept - z * se

ax.plot(osm, upper, "k--", linewidth=0.8, label="Envelope 95%")
ax.plot(osm, lower, "k--", linewidth=0.8)

ax.set_title("Normal Probability Plot - High Spot Inf Neck")
ax.set_ylabel("High Spot Inf Neck")
ax.set_xlabel("Quantis Teóricos")
ax.legend()
plt.show()
