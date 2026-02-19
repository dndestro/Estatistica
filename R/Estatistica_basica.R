library(readxl)
library(ggplot2)
library(car)
library(nortest)

#cria dados para teste do script
dados <- rnorm(1000)
dados2 <- rnorm(500)

# Resumo estatístico
summary(amostras)

# Histograma
hist(amostras, main = "Distribuição Normal", xlab = "Valores", col = "steelblue")


#gera o gráfico de fitting da distribuição
qqPlot(dados,
       distribution="norm",
       envelope = list(style = "lines", level = 0.95),
       main="Normal Probability Plot - High Spot Inf Neck",
       ylab="High Spot Inf Neck")

#Teste de Anderson-Darling
#p-value < 0,05 rejeita a normalidade
ad.test(dados)

#Teste de Kolmogorov-Smirnov
#p-value < 0,05 rejeita a normalidade
ks.test(dados, "pnorm", mean(dados), sd(dados))

#Teste deShapiro-Wilk
#p-value < 0,05 rejeita a normalidade
shapiro.test(dados)

#Premissas do teste t não pareado
#1. Independência
#As observações de um grupo não influenciam as do outro, e as amostras são coletadas de forma independente.
#2. Normalidade
#Os dados de cada grupo devem seguir uma distribuição normal. Pode ser verificada com Shapiro-Wilk, Anderson-Darling ou QQ-Plot. Para amostras grandes (n > 30), o Teorema do Limite Central relaxa essa premissa.
#3. Homogeneidade de variâncias (homocedasticidade)
#As variâncias dos dois grupos devem ser iguais. Verificada com o teste de Levene ou teste F. Caso seja violada, usa-se a versão de Welch (var.equal = FALSE no R, que é o padrão).
#4. Escala contínua
#A variável dependente deve ser medida em escala contínua (intervalar ou razão).

#teste de levene para verificar homocedasticidade
# Organizar em um dataframe
df <- data.frame(
  valores = c(dados, dados2),
  grupo   = rep(c("G1", "G2"))
)

# Teste de Levene
#p-value > 0.05 → variâncias homogêneas (não rejeita homocedasticidade)
#p-value < 0.05 → variâncias heterogêneas → use Welch (var.equal = FALSE)
leveneTest(valores ~ grupo, data = df)


#Teste t não pareado
# Se as variâncias forem iguais var.equal = TRUE, se não, var.equal = FALSE
# p-value > 0,05, Não há diferença estatisticamente significativa entre as médias dos dois grupos 
t.test(dados, dados2, var.equal = TRUE)


#Teste t pareado

#premissas do teste t pareado
#1. Pareamento
#Cada observação de um grupo deve ter um par correspondente no outro grupo 
#(ex: mesmo indivíduo antes e depois, mesmo equipamento medido por dois métodos).
#2. Normalidade das diferenças
#Não é necessário que cada grupo seja normal individualmente, mas sim que a 
#diferença entre os pares siga uma distribuição normal.

# Simular medições antes e depois
antes  <- rnorm(50, mean = 100, sd = 10)
depois <- antes + rnorm(50, mean = 5, sd = 3)  # depois é levemente maior

#testar normalidade das diferenças com algum teste de normalidade
diferenca <- antes - depois
ad.test(diferenca)

# Teste t pareado
#p-value > 0,05 indica que há forte evidÊncia para rejeitar a hipótese nula,
#ou seja, a hipótese alternativa é que a diferença média entre os pares é 
#diferente de zero (teste bilateral).
t.test(antes, depois, paired = TRUE)
