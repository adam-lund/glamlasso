library(glamlasso)

# warnings = errors
options(warn=2)

##size of example 
n1 <- 65; n2 <- 26; n3 <- 13; p1 <- 13; p2 <- 5; p3 <- 4

##marginal design matrices (Kronecker components)
X1 <- matrix(rnorm(n1 * p1), n1, p1) 
X2 <- matrix(rnorm(n2 * p2), n2, p2) 
X3 <- matrix(rnorm(n3 * p3), n3, p3) 
X <- list(X1, X2, X3)

##gaussian example 
Beta <- array(rnorm(p1 * p2 * p3) * rbinom(p1 * p2 * p3, 1, 0.1), c(p1 , p2, p3))
mu <- RH(X3, RH(X2, RH(X1, Beta)))
Y <- array(rnorm(n1 * n2 * n3, mu), dim = c(n1, n2, n3))

fit <- glamlasso(X, Y, family = "gaussian", penalty = "lasso", iwls = "exact")
Betafit <- fit$coef

modelno <- length(fit$lambda)
m <- min(Betafit[ , modelno], c(Beta))
M <- max(Betafit[ , modelno], c(Beta))
plot(c(Beta), type="l", ylim = c(m, M))
lines(Betafit[ , modelno], col = "red")

##poisson example
Beta <- array(rnorm(p1 * p2 * p3, 0, 0.1) * rbinom(p1 * p2 * p3, 1, 0.1), c(p1 , p2, p3))

mu <- RH(X3, RH(X2, RH(X1, Beta)))
Y <- array(rpois(n1 * n2 * n3, exp(mu)), dim = c(n1, n2, n3))
fit <- glamlasso(X, Y, family = "poisson", penalty = "lasso", iwls = "exact", nu = 0.1)
Betafit <- fit$coef

modelno <- length(fit$lambda)
m <- min(Betafit[ , modelno], c(Beta))
M <- max(Betafit[ , modelno], c(Beta))
plot(c(Beta), type="l", ylim = c(m, M))
lines(Betafit[ , modelno], col = "red")
