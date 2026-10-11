# =============================================================================
# Análise exploratória (EDA) + identificação da melhor distribuição estatística
# Equivalente em R do script eda_distribuicoes.py
#
# Uso:  Rscript eda_distribuicoes.R      (ou abra no RStudio e ajuste a seção CONFIG)
# =============================================================================

# ------------------------------- CONFIG --------------------------------------
arquivo <- NULL          # caminho do CSV; NULL = dados simulados (demo)
sep     <- ","           # separador do CSV
coluna  <- "valor"       # coluna numérica a analisar
grupo   <- "grupo"       # coluna categórica p/ boxplot por grupo (ou NULL)
saida   <- "saida_eda_r" # pasta de saída
top_n   <- 3             # nº de distribuições nos gráficos
n_boot  <- 0             # réplicas p/ p-valor KS por bootstrap paramétrico (0 = desliga)
alpha   <- 0.05
# -----------------------------------------------------------------------------

# Pacotes -----------------------------------------------------------------------
pkgs <- c("ggplot2", "patchwork", "fitdistrplus", "moments", "nortest", "goftest")
novos <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(novos)) install.packages(novos)
suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(fitdistrplus); library(moments)
})
# nortest e goftest são usados com `::` (ambos definem ad.test)

dir.create(saida, showWarnings = FALSE, recursive = TRUE)
options(width = 200)

# 1. Carga de dados -------------------------------------------------------------
if (is.null(arquivo)) {
  set.seed(42)
  valor <- c(rgamma(480, shape = 2.5, scale = 4), rgamma(20, shape = 2.5, scale = 4) * 3)
  df <- data.frame(valor = valor,
                   grupo = sample(c("A", "B"), length(valor), replace = TRUE),
                   outra = rnorm(length(valor), 50, 10))
  cat(">> Nenhum arquivo informado: usando dados simulados (coluna 'valor').\n\n")
} else {
  df <- read.csv(arquivo, sep = sep)
  if (!coluna %in% names(df)) stop("Coluna '", coluna, "' não existe no arquivo.")
}
if (!is.null(grupo) && !grupo %in% names(df)) grupo <- NULL

# 2. EDA numérica ---------------------------------------------------------------
cat(strrep("=", 80), "\nVISÃO GERAL\n", strrep("=", 80), "\n", sep = "")
cat("Dimensão:", nrow(df), "linhas x", ncol(df), "colunas\n\n")
str(df)
cat("\nResumo:\n"); print(summary(df))
cat("\nValores ausentes por coluna:\n"); print(colSums(is.na(df)))

x <- as.numeric(na.omit(df[[coluna]]))
n <- length(x)

cat("\n", strrep("=", 80), "\nVARIÁVEL ANALISADA: ", coluna, "\n", strrep("=", 80), "\n", sep = "")
print(quantile(x, c(0, .01, .05, .25, .5, .75, .95, .99, 1)))
cat(sprintf("\nMédia: %.4f | DP: %.4f | CV: %.4f\n", mean(x), sd(x), sd(x) / mean(x)))
cat(sprintf("Assimetria: %.4f | Curtose (excesso): %.4f\n", skewness(x), kurtosis(x) - 3))

q   <- quantile(x, c(.25, .75)); iqr <- diff(q)
li  <- q[1] - 1.5 * iqr;         ls  <- q[2] + 1.5 * iqr
out <- x[x < li | x > ls]
cat(sprintf("\nOutliers (Tukey 1.5*IQR): %d (%.1f%%) fora de [%.3f, %.3f]\n",
            length(out), 100 * length(out) / n, li, ls))

num <- df[sapply(df, is.numeric)]
if (ncol(num) > 1) { cat("\nCorrelação (Pearson):\n"); print(round(cor(num, use = "complete.obs"), 3)) }

# Testes de normalidade (H0: dados ~ Normal)
cat("\n", strrep("=", 80), "\nTESTES DE NORMALIDADE (H0: dados ~ Normal)\n", strrep("=", 80), "\n", sep = "")
amostra <- if (n <= 5000) x else sample(x, 5000)
norm_tab <- data.frame(
  teste = c("Shapiro-Wilk", "D'Agostino (assimetria)", "Jarque-Bera",
            "Lilliefors (KS)", "Anderson-Darling"),
  p_valor = c(shapiro.test(amostra)$p.value,
              agostino.test(x)$p.value,
              jarque.test(x)$p.value,
              nortest::lillie.test(x)$p.value,
              nortest::ad.test(x)$p.value)
)
norm_tab$decisao <- ifelse(norm_tab$p_valor < alpha, "rejeita H0", "não rejeita H0")
print(norm_tab, row.names = FALSE, digits = 4)

# 3. Gráficos de EDA (inclui boxplot) -------------------------------------------
d <- data.frame(x = x)
p1 <- ggplot(d, aes(x)) + geom_histogram(aes(y = after_stat(density)), bins = 30,
        fill = "steelblue", alpha = .6, color = "white") +
      geom_density(linewidth = 1) + labs(title = "Histograma + densidade (KDE)", x = coluna)
p2 <- ggplot(d, aes(x = "", y = x)) + geom_boxplot(fill = "lightsteelblue", outlier.colour = "red") +
      geom_jitter(width = .15, size = .6, alpha = .25) + coord_flip() +
      labs(title = "Boxplot (com pontos individuais)", x = NULL, y = coluna)
p3 <- ggplot(d, aes(x = "", y = x)) + geom_violin(fill = "lightcoral", draw_quantiles = c(.25, .5, .75)) +
      coord_flip() + labs(title = "Violin plot", x = NULL, y = coluna)
p4 <- ggplot(d, aes(x)) + stat_ecdf(geom = "step") + labs(title = "ECDF", x = coluna, y = "F(x)")
ggsave(file.path(saida, "01_eda_variavel.png"), (p1 | p2) / (p3 | p4), width = 13, height = 9, dpi = 150)

if (!is.null(grupo)) {
  pg <- ggplot(df, aes(x = factor(.data[[grupo]]), y = .data[[coluna]], fill = factor(.data[[grupo]]))) +
    geom_boxplot(show.legend = FALSE) + geom_jitter(width = .15, size = .6, alpha = .3) +
    labs(title = paste("Boxplot de", coluna, "por", grupo), x = grupo, y = coluna)
  ggsave(file.path(saida, "02_boxplot_por_grupo.png"), pg, width = 8, height = 5, dpi = 150)
}
if (ncol(num) > 1) {
  png(file.path(saida, "03_boxplots_numericas.png"), width = 400 * ncol(num), height = 550, res = 120)
  par(mfrow = c(1, ncol(num)))
  for (v in names(num)) boxplot(num[[v]], main = v, col = "lightsteelblue")
  dev.off()
}

# 4. Ajuste de distribuições + testes de aderência -------------------------------
cat("\n", strrep("=", 80), "\nAJUSTE DE DISTRIBUIÇÕES (MLE) E TESTES DE ADERÊNCIA\n", strrep("=", 80), "\n", sep = "")

candidatas <- c("norm", "lnorm", "gamma", "weibull", "exp", "logis")
if (min(x) <= 0) candidatas <- setdiff(candidatas, c("lnorm", "gamma", "weibull", "exp"))

ajustes <- lapply(candidatas, function(dd)
  tryCatch(suppressWarnings(fitdist(x, dd)), error = function(e) NULL))
names(ajustes) <- candidatas
ajustes <- Filter(Negate(is.null), ajustes)

ks_stat <- function(dados, dd, est) {
  unname(suppressWarnings(do.call(ks.test, c(list(dados, paste0("p", dd)), as.list(est))))$statistic)
}

# p-valor KS por bootstrap paramétrico (p-valor clássico é otimista c/ parâmetros estimados)
ks_bootstrap <- function(fit, dados, B) {
  dd <- fit$distname; est <- fit$estimate
  obs <- ks_stat(dados, dd, est)
  rfun <- get(paste0("r", dd))
  cont <- 0; validos <- 0
  for (b in seq_len(B)) {
    amostra <- do.call(rfun, c(list(length(dados)), as.list(est)))
    f <- tryCatch(suppressWarnings(fitdist(amostra, dd)), error = function(e) NULL)
    if (is.null(f)) next
    validos <- validos + 1
    cont <- cont + (ks_stat(amostra, dd, f$estimate) >= obs)
  }
  (cont + 1) / (validos + 1)
}

tab <- do.call(rbind, lapply(names(ajustes), function(dd) {
  f    <- ajustes[[dd]]
  args <- as.list(f$estimate)
  ks   <- suppressWarnings(do.call(ks.test, c(list(x, paste0("p", dd)), args)))
  cvm  <- do.call(goftest::cvm.test, c(list(x, null = paste0("p", dd)), args))
  ad   <- do.call(goftest::ad.test,  c(list(x, null = paste0("p", dd)), args))
  data.frame(distribuicao = dd, k = length(f$estimate), loglik = f$loglik,
             AIC = f$aic, BIC = f$bic,
             KS = unname(ks$statistic), KS_p = ks$p.value,
             CvM = unname(cvm$statistic), CvM_p = cvm$p.value,
             AD = unname(ad$statistic), AD_p = ad$p.value)
}))
tab <- tab[order(tab$AIC), ]
rownames(tab) <- NULL
tab$dAIC        <- tab$AIC - min(tab$AIC)
tab$peso_Akaike <- exp(-0.5 * tab$dAIC) / sum(exp(-0.5 * tab$dAIC))

tab$KS_p_boot <- NA_real_
if (n_boot > 0) {
  cat(sprintf("\nBootstrap paramétrico (%d réplicas) para as 3 melhores por AIC...\n", n_boot))
  for (i in seq_len(min(3, nrow(tab))))
    tab$KS_p_boot[i] <- ks_bootstrap(ajustes[[tab$distribuicao[i]]], x, n_boot)
}
tab$aceita_KS <- ifelse(ifelse(is.na(tab$KS_p_boot), tab$KS_p, tab$KS_p_boot) > alpha, "sim", "não")

print(tab[, c("distribuicao", "k", "AIC", "BIC", "dAIC", "peso_Akaike",
              "KS", "KS_p", "KS_p_boot", "CvM", "CvM_p", "AD", "AD_p", "aceita_KS")],
      digits = 4, row.names = FALSE)
write.csv(tab, file.path(saida, "ranking_distribuicoes.csv"), row.names = FALSE)

# Análise gráfica: PDF, CDF, Q-Q e P-P das melhores distribuições
top      <- ajustes[tab$distribuicao[seq_len(min(top_n, nrow(tab)))]]
legendas <- names(top)

png(file.path(saida, "05_ajuste_diagnosticos.png"), width = 1800, height = 1500, res = 150)
par(mfrow = c(2, 2))
denscomp(top, legendtext = legendas, main = "Histograma + PDFs ajustadas")
cdfcomp(top,  legendtext = legendas, main = "ECDF vs CDFs teóricas")
qqcomp(top,   legendtext = legendas, main = "Q-Q plot")
ppcomp(top,   legendtext = legendas, main = "P-P plot")
dev.off()

# Gráfico de Cullen-Frey (assimetria² x curtose) com bootstrap — guia visual
png(file.path(saida, "06_cullen_frey.png"), width = 1000, height = 900, res = 150)
descdist(x, boot = 500)
dev.off()

# Q-Q normal clássico (referência)
png(file.path(saida, "07_qq_normal.png"), width = 900, height = 900, res = 150)
qqnorm(x, main = "Q-Q plot Normal (referência)"); qqline(x, col = "red", lty = 2)
dev.off()

melhor <- tab$distribuicao[1]
cat(sprintf("\n>> Melhor por AIC: %s | parâmetros (R): %s\n", melhor,
            paste(names(ajustes[[melhor]]$estimate),
                  round(ajustes[[melhor]]$estimate, 4), sep = "=", collapse = ", ")))
cat(">> Gráficos e tabela salvos em:", normalizePath(saida), "\n")
cat("
NOTAS DE INTERPRETAÇÃO
 - AIC/BIC comparam distribuições entre si (menor = melhor); dAIC < 2 = modelos equivalentes.
 - KS, Cramér-von Mises e Anderson-Darling avaliam aderência absoluta (H0: dados ~ distribuição).
   Com parâmetros estimados dos próprios dados os p-valores clássicos são otimistas:
   use n_boot > 0 para um p-valor KS por bootstrap paramétrico.
 - Em amostras grandes qualquer teste rejeita desvios minúsculos; confie também nos gráficos.
")
