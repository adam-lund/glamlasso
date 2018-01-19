#
#     Description of this R script:
#     R interface/wrapper for the Rcpp function gdpg in the glamlasso package.
#
#     Intended for use with R.
#     Copyright (C) 2016 Adam Lund
# 
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
# 
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>
#

#' @name glamlasso
#' 
#' @title Penalization in Large Scale Generalized Linear Array Models
#' 
#' @description  Efficient design matrix free procedure for fitting large scale penalized  2 or 3-dimensional
#' generalized linear array models. Currently the LASSO penalty and the SCAD penalty are both implemented. Furthermore,
#' the Gaussian model with identity link,  the Binomial model with logit link, the Poisson model
#' with log link and the Gamma model with log link is currently implemented. The \code{glamlasso} function
#' utilize a gradient descent and proximal gradient based algorithm, see  \cite{Lund et al., 2015}. 
#'  
#' @usage  glamlasso(X, 
#'           Y, 
#'           family = c("gaussian", "binomial", "poisson", "gamma"),
#'           penalty = c("lasso", "scad"),
#'           Weights = NULL,
#'           nlambda = 100,
#'           lambda.min.ratio = 1e-04,
#'           lambda = NULL,
#'           penalty.factor = NULL,
#'           reltolinner = 1e-07,
#'           reltolouter = 1e-04,
#'           maxiter = 15000,
#'           steps = 1,
#'           maxiterinner = 3000,
#'           maxiterouter = 25,
#'           btinnermax = 100,
#'           iwls = c("exact", "identity", "kron1", "kron2"),
#'           nu = 1)
#'            
#' @param X A list containing the Kronecker components (2 or 3) of the Kronecker design matrix.
#'  These are  matrices of sizes \eqn{n_i   \times p_i}.
#' @param Y The response values, an array of size \eqn{n_1 \times\cdots\times n_d}. For option 
#' \code{family = "binomial"} this array must contain the proportion of successes and the 
#' number of trials is then specified as \code{Weights} (see below).
#' @param family A string specifying the model family (essentially the response distribution). Possible values 
#' are \code{"gaussian", "binomial", "poisson", "gamma"}.
#' @param penalty A string specifying the penalty. Possible values 
#' are \code{"lasso", "scad"}.
#' @param Weights Observation weights, an array of size \eqn{n_1 \times \cdots \times n_d}. For option 
#' \code{family = "binomial"} this array must contain the number of trials and must be provided.
#' @param nlambda The number of \code{lambda} values.
#' @param lambda.min.ratio The smallest value for \code{lambda}, given as a fraction of 
#' \eqn{\lambda_{max}}; the (data derived) smallest value for which all coefficients are zero.
#' @param lambda The sequence of penalty parameters for the regularization path.
#' @param penalty.factor An array of size \eqn{p_1 \times \cdots \times p_d}. Is multiplied 
#' with each element in \code{lambda} to allow differential shrinkage on the coefficients. 
#' @param reltolinner The convergence tolerance for the inner loop
#' @param reltolouter The convergence tolerance for the outer loop.
#' @param maxiter The maximum number of inner iterations allowed for each \code{lambda}
#' value, when  summing over all outer iterations for said \code{lambda}.
#' @param steps The number of steps used in the multi-step adaptive lasso algorithm for non-convex penalties. Automatically set to 1 when \code{penalty = "lasso"}.
#' @param maxiterinner The maximum number of inner iterations allowed for each outer iteration.
#' @param maxiterouter The maximum number of outer iterations allowed for each lambda.
#' @param btinnermax Maximum number of backtracking steps allowed in each inner iteration. Default is \code{btinnermax = 100}.
#' @param iwls A string indicating whether to use the exact iwls weight matrix or use a kronecker structured approximation to it.
#' @param nu A number between 0 and 1 that controls the step size \eqn{\delta} in the proximal algorithm (inner loop) by 
#' scaling the upper bound \eqn{\hat{L}_h} on the Lipschitz constant \eqn{L_h} (see \cite{Lund et al., 2015}). 
#' For \code{nu = 1} backtracking never occurs and the proximal step size is always \eqn{\delta = 1 / \hat{L}_h}. 
#' For \code{nu = 0} backtracking always occurs and the proximal step size is initially \eqn{\delta = 1}. 
#' For \code{0 < nu < 1} the proximal step size is initially \eqn{\delta = 1/(\nu\hat{L}_h)} and backtracking 
#' is only employed if the objective function does not decrease. A \code{nu} close  to 0 gives large step 
#' sizes and presumably more backtracking in the inner loop. The default is \code{nu = 1} and the option is only 
#' used if \code{iwls = "exact"}.
#' 
#' @details We consider a generalized linear model (GLM) with Kronecker structured design matrix given as
#'  \deqn{X = \bigotimes_{i=1}^d X_i,}
#' where \eqn{X_1,X_2,\ldots} are the marginal \eqn{n_i\times p_i} design matrices (Kronecker components). 
#'  
#' We use the generalized linear array model (GLAM) framework to write the model equation as
#'  \deqn{g(M) = \rho(X_d,\rho(X_{d-1},\ldots,\rho(X_1,\Theta))),}
#' where \eqn{\rho} is the so called rotated \eqn{H}-transfrom, \eqn{M} is an array containing the mean of the 
#' response variable array \eqn{Y}, \eqn{\Theta} is the  model coefficient (parameter) array and \eqn{g} is a link function. 
#'  See \cite{Currie et al., 2006} for more details.
#'         
#' Let \eqn{\theta : = vec(\Theta)} denote the vectorized version of the parameter array. The related log-likelihood is a function of 
#' \eqn{\theta} through the linear predictor \eqn{\eta} i.e. \eqn{\theta \mapsto l(\eta(\theta))}.
#' In the usual exponential family framework this can be expressed as
#' \deqn{l(\eta(\theta)) = \sum_{i = 1}^n a_i \frac{y_i \vartheta(\eta_i(\theta)) - b(\vartheta(\eta_i(\theta)))}{\psi}+c(y_i,\psi)} 
#' where \eqn{\vartheta}, the canonical parameter map,  is linked to the  linear predictor via the identity
#'  \eqn{\eta(\theta) = g(b'(\vartheta))} with \eqn{b} the cumulant function. Here \eqn{a_i \ge 0, i = 1,\ldots,n} are observation weights and
#'  \eqn{\psi} is the dispersion parameter.
#' 
#' Using only the marginal matrices \eqn{X_1,X_2,\ldots} and with \eqn{J} a penalty function, the function \code{glamlasso} solves the penalized estimation problem 
#' \deqn{\min_{\theta} -l(\eta(\theta)) + \lambda J (\theta),} 
#'in the GLAM setup for a sequence of penalty parameters \eqn{\lambda>0}. The underlying algorithm is based on an outer 
#' gradient descent loop and an inner proximal gradient based loop. We note that if \eqn{J} is not 
#' convex, as with the SCAD penalty, we use the multiple step adaptive lasso procedure to loop over the inner proximal algorithm, see \cite{Lund et al., 2015} for more details.
#'   
#' @return An object with S3 Class "glamlasso". 
#' \item{spec}{A string indicating the GLAM dimension (2 or 3), the model family and the penalty.}  
#' \item{coef}{A \eqn{p_1\cdots p_d \times} \code{nlambda} matrix containing the estimates of 
#' the model coefficients (\code{beta}) for each \code{lambda}-value.}
#' \item{lambda}{A vector containing the sequence of penalty values used in the estimation procedure.}
#' \item{df}{The number of nonzero coefficients for each value of \code{lambda}.}	
#' \item{dimcoef}{A vector giving the dimension of the model coefficient array \eqn{\beta}.}
#' \item{dimobs}{A vector giving the dimension of the observation (response) array \code{Y}.}
#' \item{Iter}{A list with 4 items:  
#' \code{bt_iter_inner}  is total number of backtracking steps performed in the inner loop,
#' \code{bt_enter_inner} is the number of times the backtracking is initiated in the inner loop,
#' \code{bt_iter_outer} is total number of backtracking steps performed in the outer loop,
#' and \code{iter_mat} is a \code{nlambda} \eqn{\times} \code{maxiterouter} matrix containing the  number of 
#' inner iterations for each \code{lambda} value and each outer iteration and  \code{iter} is total number of iterations i.e. \code{sum(Iter)}.}  
#'  
#' @author  Adam Lund
#' 
#' Maintainer: Adam Lund, \email{adam.lund@@math.ku.dk}
#' 
#' @references 
#' Lund, A., M. Vincent, and N. R. Hansen (2015). Penalized estimation in 
#' large-scale generalized linear array models. 
#' \emph{ArXiv}. 
#' 
#' Currie, I. D., M. Durban, and P. H. C. Eilers (2006). Generalized linear
#' array models with applications to multidimensional smoothing. 
#' \emph{Journal of the Royal Statistical Society. Series B}. 68, 259-280.
#' 
#' @keywords package 
#'
#' @examples 
#' \dontrun{
#' ##size of example 
#' n1 <- 65; n2 <- 26; n3 <- 13; p1 <- 13; p2 <- 5; p3 <- 4
#' 
#' ##marginal design matrices (Kronecker components)
#' X1 <- matrix(rnorm(n1 * p1), n1, p1) 
#' X2 <- matrix(rnorm(n2 * p2), n2, p2) 
#' X3 <- matrix(rnorm(n3 * p3), n3, p3) 
#' X <- list(X1, X2, X3)
#' 
#' ##gaussian example 
#' Beta <- array(rnorm(p1 * p2 * p3) * rbinom(p1 * p2 * p3, 1, 0.1), c(p1 , p2, p3))
#' mu <- RH(X3, RH(X2, RH(X1, Beta)))
#' Y <- array(rnorm(n1 * n2 * n3, mu), dim = c(n1, n2, n3))
#' 
#' fit <- glamlasso(X, Y, family = "gaussian", penalty = "lasso", iwls = "exact")
#' Betafit <- fit$coef
#' 
#' modelno <- length(fit$lambda)
#' m <- min(Betafit[ , modelno], c(Beta))
#' M <- max(Betafit[ , modelno], c(Beta))
#' plot(c(Beta), type="l", ylim = c(m, M))
#' lines(Betafit[ , modelno], col = "red")
#' 
#' ##poisson example
#' Beta <- array(rnorm(p1 * p2 * p3, 0, 0.1) * rbinom(p1 * p2 * p3, 1, 0.1), c(p1 , p2, p3))
#' 
#' mu <- RH(X3, RH(X2, RH(X1, Beta)))
#' Y <- array(rpois(n1 * n2 * n3, exp(mu)), dim = c(n1, n2, n3))
#' fit <- glamlasso(X, Y, family = "poisson", penalty = "lasso", iwls = "exact", nu = 0.1)
#' Betafit <- fit$coef
#' 
#' modelno <- length(fit$lambda)
#' m <- min(Betafit[ , modelno], c(Beta))
#' M <- max(Betafit[ , modelno], c(Beta))
#' plot(c(Beta), type="l", ylim = c(m, M))
#' lines(Betafit[ , modelno], col = "red")
#' }

glamlasso <-function(X,
                     Y, 
                     family = c("gaussian", "binomial", "poisson", "gamma"),
                     penalty = c("lasso", "scad"),
                     Weights = NULL,
                     nlambda = 100,
                     lambda.min.ratio = 0.0001,
                     lambda = NULL,
                     penalty.factor = NULL,
                     reltolinner = 1e-07,
                     reltolouter = 1e-04,
                     maxiter = 15000,
                     steps = 1,
                     maxiterinner = 3000,
                     maxiterouter = 25,
                     btinnermax = 100,
                     iwls = c("exact", "identity", "kron1", "kron2"),
                     nu = 1) {
  
##get dimensions of problem
dimglam <- length(X)

if (dimglam < 2 || dimglam > 3){
  
stop(paste("the dimension of the GLAM must be 2 or 3!"))
  
}else if (dimglam == 2){X[[3]] <- matrix(1, 1, 1)} 

X1 <- X[[1]]
X2 <- X[[2]]
X3 <- X[[3]]

dimX <- rbind(dim(X1), dim(X2), dim(X3))

n1 <- dimX[1, 1]
n2 <- dimX[2, 1]
n3 <- dimX[3, 1]
p1 <- dimX[1, 2]
p2 <- dimX[2, 2]
p3 <- dimX[3, 2]
n <- prod(dimX[,1])
p <- prod(dimX[,2])

####reshape Y into matrix
Y <- matrix(Y, n1, n2 * n3)

####check on family
if(sum(family == c("gaussian", "binomial", "poisson", "gamma")) != 1){

stop(paste("family must be correctly specified"))

}
 
####check on penalty
if(sum(penalty == c("lasso", "scad")) != 1){stop(paste("penalty must be correctly specified"))}

if(penalty == "lasso"){steps <- 1}

####check on weights 
if(family == "binomial" & is.null(Weights)){stop(paste("for binomial model number of trials (Weights) must be specified"))}

if(is.null(Weights)){
  
weightedgaussian <- 0  
Weights <- matrix(1, n1, n2 * n3)

}else{

if(min(Weights) < 0){stop(paste("only positive weights allowed"))}    

weightedgaussian <- 1
Weights <- matrix(Weights, n1, n2 * n3)

}

####check on lambda 
if(is.null(lambda)){
  
makelamb <- 1
lambda <- rep(NA, nlambda)
  
}else{
  
if(length(lambda) != nlambda){

stop(
paste("number of elements in lambda (", length(lambda),") is not equal to nlambda (", nlambda,")", sep = "")
)

}
  
makelamb <- 0

}  

####check on penalty.factor
if(is.null(penalty.factor)){
  
penalty.factor <- matrix(1, p1, p2 * p3)
  
}else if(prod(dim(penalty.factor)) != p){
  
stop(
paste("number of elements in penalty.factor (", length(penalty.factor),") is not equal to the number of coefficients (", p,")", sep = "")
)
  
}else {
  
if(min(penalty.factor) < 0){stop(paste("penalty.factor must be positive"))}    
  
penalty.factor <- matrix(penalty.factor, p1, p2 * p3)
  
}  

####check on iwls
if(sum(iwls == c("exact", "identity", "kron1", "kron2")) != 1){stop(paste("iwls must be correctly specified"))}

####check on nu
if(nu < 0 || nu > 1){stop(paste("nu must be between 0 and 1"))}

##run gdpg algorithm
res <- gdpg(X1, X2, X3,
            Y, 
            Weights, 
            family,  
            penalty,
            iwls, 
            nu,
            lambda, makelamb, nlambda, lambda.min.ratio,
            penalty.factor,
            reltolinner, 
            reltolouter, 
            maxiter,
            steps,
            maxiterinner,
            maxiterouter,
            btinnermax,
            weightedgaussian)

####checks
if(res$STOPmaxiter == 1){

warning(paste("program exit due to maximum number of iterations (",maxiter,") reached for model no ",res$endmodelno,""))

}

if(res$STOPprox == 1){

warning(paste("program exit due to maximum number of backtraking steps in the inner loop reached for model no ",res$endmodelno,""))

}

if(res$STOPnewt == 1){

warning(paste("program exit due to max number of backtraking steps in the outer loop reached for model no ",res$endmodelno,""))

}

endmodelno <- res$endmodelno
Iter <- res$Iter

if(family != "gaussian" || (weightedgaussian == 1 & iwls == "kron")){
  
maxiterouterreached <- ifelse(Iter[1:endmodelno, maxiterouter] > 0, 1, 0) * (1:endmodelno)
maxiterouterreached <- maxiterouterreached[maxiterouterreached > 0]

if(length(maxiterouterreached)){

models <- paste("", maxiterouterreached)
warning(paste("maximum number of outer iterations (", maxiterouter,") reached for model(s):"), models)

}

}

maxiterinnerpossible <- sum(Iter > 0)
maxiterinnerreached <- sum(Iter >= (maxiterinner - 1)) 

if(maxiterinnerreached > 0){

warning(
paste("maximum number of inner iterations (",maxiterinner,") reached ",maxiterinnerreached," time(s) out of ",maxiterinnerpossible," possible")
)

}

out <- list()

class(out) <- "glamlasso"

out$spec <- paste("", dimglam,"-dimensional ", penalty," penalized ", family," GLAM") 
out$coef <- res$Beta[ , 1:endmodelno]
out$lambda <- res$lambda[1:endmodelno] 
out$df <- res$df[1:endmodelno]
out$dimcoef <- c(p1, p2, p3)[1:dimglam]
out$dimobs <- c(n1, n2, n3)[1:dimglam]

Iter <- list()
Iter$bt_enter_inner <- res$btenterprox
Iter$bt_iter_inner <- res$btiterprox
Iter$bt_iter_outer <- res$btiternewt
Iter$iter_mat <- res$Iter[1:endmodelno, ]
Iter$iter <- sum(Iter$iter_mat, na.rm = TRUE)

out$Iter <- Iter

return(out)

}