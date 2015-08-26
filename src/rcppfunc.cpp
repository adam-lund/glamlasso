/*  
    Two Rcpp functions used in connection with LASSO regularization for 
    generalized linear array models (GLAM).
    The first function, pgal, contains a proximal gradient based IWLS algorithm that solves
    the LASSO problem in the GLAM framework.
    The second function, getobj, computes the objective values for this LASSO problem.

    Intended for use with R.
    Copyright (C) 2015 Adam Lund

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>
*/

//// [[Rcpp::depends(RcppArmadillo)]]
//#define TIMING
#include <RcppArmadillo.h>
#include "auxfunc.h"
//#include "/Users/adamlund/Documents/KU/Phd/Project/Computer/Vincent/timer/simple_timer.h"

using namespace std;
using namespace arma;

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// cgd fpg algorithm ///////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//[[Rcpp::export]]
Rcpp::List pgal(arma::mat Phi1, arma::mat Phi2, arma::mat Phi3,
                arma::mat Y, arma::mat Weights,
                std::string family,
                std::string iwls, 
                double nu,
                arma::vec lambda, int makelamb,  int nlambda, double  lambdaminratio,
                arma::mat penaltyfactor,
                double reltolprox,
                double reltolnewt,
                int maxiter,
                int maxiterprox,
                int maxiternewt,
                int btproxmax,
                int weightedgaussian){
                  
Rcpp::List output;

//declare some global variables
int p1 = Phi1.n_cols;
int p2 = Phi2.n_cols;
int p3 = Phi3.n_cols;
int p = p1 * p2 * p3;
int n1 = Phi1.n_rows;
int n2 = Phi2.n_rows;
int n3 = Phi3.n_rows;
int n = n1 * n2 * n3;
        
int btenterprox = 0, btiternewt = 0, btiterprox = 0,
    endmodelno = nlambda,
    STOPmaxiter = 0,  STOPnewt = 0, STOPprox = 0;

if(family == "gaussian"){//gaussian#################################################

if(weightedgaussian == 0){
  
////declare variables
double delta, 
       L,  
       relobjprox,
       sqlossBeta, sqlossProp;

arma::vec df(nlambda), 
          eig1, eig2, eig3,
          Iter(nlambda),
          obj(maxiterprox + 1);

arma::mat Beta(p1, p2 * p3), Betaprev(p1, p2 * p3), Betas(p, nlambda), 
          Gamma(p1, p2 * p3), GradsqlossX(p1, p2 * p3),  
          Prop(p1, p2 * p3), Phi1tPhi1, Phi2tPhi2, Phi3tPhi3,  PhitY, 
          X(p1, p2 * p3);
 
////fill variables 
obj.fill(0);

Iter.fill(0);

////precompute 
Phi1tPhi1 = Phi1.t() * Phi1;
Phi2tPhi2 = Phi2.t() * Phi2;
Phi3tPhi3 = Phi3.t() * Phi3;
PhitY = RHmat(Phi3.t(), RHmat(Phi2.t(), RHmat(Phi1.t(), Y, n2, n3), n3, p1), p1, p2);

eig1 = arma::eig_sym(Phi1tPhi1);
eig2 = arma::eig_sym(Phi2tPhi2);
eig3 = arma::eig_sym(Phi3tPhi3);
L = as_scalar(max(kron(eig1, kron(eig2 , eig3)))) / n;
delta = 1 / L;
 
////initialize
Beta.fill(0);
sqlossBeta = sqloss(Phi1, Phi2, Phi3, Y, Beta, n, p2, p3, n1, n2);

////make lambda sequence        
if(makelamb == 1){
  
double lambdamax = as_scalar(max(max(abs(PhitY) / penaltyfactor))) / n;
double m = log(lambdaminratio);
double M = 0;
double difflamb = abs(M - m) / (nlambda - 1);
double l = 0;

for(int i = 0; i < nlambda ; i++){
  
lambda(i) = lambdamax * exp(l);
l = l - difflamb;

}

}else{std::sort(lambda.begin(), lambda.end(), std::greater<int>());}
        
////lambda loop
for (int j = 0; j < nlambda; j++){
  
Gamma = penaltyfactor * lambda(j);

////proximal loop
for (int k = 0; k < maxiterprox; k++){
  
if(k == 0){
 
obj(0) = sqlossBeta + penalty(Gamma, Beta);
Betaprev = Beta;

}else{
  
X = Beta + (k - 2) / (k + 1) * (Beta - Betaprev);
GradsqlossX = (RHmat(Phi3tPhi3, RHmat(Phi2tPhi2, RHmat(Phi1tPhi1, X, p2, p3), p3, p1), p1, p2) - PhitY) / n;
Prop = prox_l1(X - delta * GradsqlossX, delta * Gamma);
sqlossProp = sqloss(Phi1, Phi2, Phi3, Y, Prop, n, p2, p3, n1, n2);

Betaprev = Beta;
Beta = Prop;
sqlossBeta = sqlossProp;

}
                
Iter(j) = k + 1;
obj(k + 1) = sqlossBeta + penalty(Gamma, Beta);

////proximal convergence check //fista not descent!
relobjprox = abs(obj(k + 1) - obj(k)) / abs(obj(k)); 

if(k > 0 && k < maxiterprox &&  relobjprox < reltolprox){//go to next lambda

obj.fill(0);
break;

}else if(k == maxiterprox){//go to next lambda

obj.fill(0);
break;

}

} //end proximal loop

df(j) = p - accu((Beta == 0));
Betas.col(j) = vectorise(Beta);

//check if maximum number of iterations for current lambda is reached
if(accu(Iter) > maxiter){STOPmaxiter = 1;}

//stop program if maxiter is reached
if(STOPmaxiter == 1){

endmodelno = j;
break;

}

} //end lambda loop

output = Rcpp::List::create(Rcpp::Named("Beta") = Betas,
                            Rcpp::Named("df") = df,
                            Rcpp::Named("endmodelno") = endmodelno,
                            Rcpp::Named("Iter") = Iter,
                            Rcpp::Named("lambda") = lambda,
                            Rcpp::Named("STOPmaxiter") = STOPmaxiter,
                            Rcpp::Named("STOPnewt") = STOPnewt,
                            Rcpp::Named("STOPprox") = STOPprox
                            );

}else if (weightedgaussian == 1){// if weights are used// solve (one) weighted ls problem

////declare variables
int ascentprox, ascentproxmax,
    btprox;
    
double alphamax,
       delta, deltamin,
       Lmax,  
       relobjprox, 
       sprox,
       tprox, 
       valprox, 
       wmax, wsqlossBeta, wsqlossProp, wsqlossX;  
       
arma::vec df(nlambda), 
          eig1, eig2, eig3, 
          Iter(nlambda), 
          objprox(maxiterprox + 1);

arma::mat Beta(p1, p2 * p3), Betaprevprox(p1, p2 * p3), Betas(p, nlambda), BTprox(nlambda, maxiterprox + 1),  
          Eta(n1, n2 * n3), 
          Gamma(p1, p2 * p3), GradwsqlossX(p1, p2 * p3),  
          MuEta(n1, n2 * n3), 
          Phi1tPhi1, Phi2tPhi2, Phi3tPhi3, PhitWZ, Prop(p1, p2 * p3), 
          SqrtW, SqrtWZ,  
          W(n1, n2 * n3),
          X(p1, p2 * p3), 
          Z(n1, n2 * n3);

////fill variables
ascentproxmax = 4;

sprox = 0.9;       
       
objprox.fill(NA_REAL);

Betas.fill(NA_REAL);
Iter.fill(0);

BTprox.fill(-1);

////precompute
Phi1tPhi1 = Phi1.t() * Phi1;
Phi2tPhi2 = Phi2.t() * Phi2;
Phi3tPhi3 = Phi3.t() * Phi3;
eig1 = arma::eig_sym(Phi1tPhi1);
eig2 = arma::eig_sym(Phi2tPhi2);
eig3 = arma::eig_sym(Phi3tPhi3);
alphamax = as_scalar(max(kron(eig1, kron(eig2 , eig3))));

////prior weight matrix
W = Weights;                 
Z = Y;

////precompute
SqrtW = sqrt(W);
SqrtWZ = SqrtW % Z;
PhitWZ = RHmat(Phi3.t(), RHmat(Phi2.t(), RHmat(Phi1.t(), W % Z, n2, n3), n3, p1), p1, p2);

////proximal step size
wmax = max(max(W));
Lmax = alphamax * wmax / n; //upper bound on Lipschitz
deltamin = 1.99 / Lmax; //minimum stepsize

//initial step size
if(nu > 0){delta = 1.9 / (nu * Lmax);}else{delta = 1;}

////initialize 
Betaprevprox.fill(0); //initialize at 0
Beta = Betaprevprox;
Eta =  RHmat(Phi3, RHmat(Phi2, RHmat(Phi1, Beta, p2, p3), p3, n1), n1, n2);
MuEta = mu(Eta, family);

////make lambda sequence
if(makelamb == 1){
  
double lambdamax = as_scalar(max(max(abs(gradloglike(Y, W, Phi1, Phi2, Phi3, MuEta, Eta, 
                                                     n2, n3, p1, p2, n, family)) / penaltyfactor)));
double m = log(lambdaminratio);
double M = 0;
double difflamb = abs(M - m) / (nlambda - 1);
double l = 0;

for(int i = 0; i < nlambda ; i++){
  
lambda(i) = lambdamax * exp(l);
l = l - difflamb;

}

}else{std::sort(lambda.begin(), lambda.end(), std::greater<int>());}
    
//lambda loop
for (int j = 0; j < nlambda; j++){
  
Gamma = penaltyfactor * lambda(j);

ascentprox = 0;
            
/////proximal loop
for (int k = 0; k < maxiterprox; k++){
  
if(k == 0){

Betaprevprox = Beta;
objprox(0) = wsqloss(SqrtW, Phi1, Phi2, Phi3, SqrtWZ, Beta, n, p2, p3, n1, n2) + penalty(Gamma, Beta);
BTprox(j, k) = 1; //force initial backtracking (if deltamin < delta)

}else{

X = Beta + (k - 2) / (k + 1) * (Beta - Betaprevprox);
GradwsqlossX = (winprod(W, Phi1, Phi2, Phi3, X, n1, n2, n3, p1, p2, p3) - PhitWZ) / n;

////check if proximal backtracking occurred last iteration
if(BTprox(j, k - 1) > 0){btprox = 1;}else{btprox = 0;}

////check for divergence
if(ascentprox > ascentproxmax){btprox = 1;}

if((btprox == 1 & deltamin < delta) || nu == 0){//backtrack
                        
wsqlossX = wsqloss(SqrtW, Phi1, Phi2, Phi3, SqrtWZ, X, n, p2, p3, n1, n2);

////proximal backtracking line search
BTprox(j, k) = 0;

while (BTprox(j, k) < btproxmax){

Prop = prox_l1(X - delta * GradwsqlossX, delta * Gamma);
wsqlossProp = wsqloss(SqrtW, Phi1, Phi2, Phi3, SqrtWZ, Prop, n, p2, p3, n1, n2);
valprox = as_scalar(wsqlossX + accu(GradwsqlossX % (Prop - X)) + 1 / (2 * delta) * sum_square(Prop - X));

if (wsqlossProp <= valprox + 0.0000001){ //need to add a little due to numerical issues

break;

}else{

delta = sprox * delta;
BTprox(j, k) = BTprox(j, k) + 1;

if(delta < deltamin){delta = deltamin;}

}

}

////check if maximum number of proximal backtraking step is reached
if(BTprox(j, k) == btproxmax){STOPprox = 1;}

}else{//no backtracking

Prop = prox_l1(X - delta * GradwsqlossX, delta * Gamma);
wsqlossProp = wsqloss(SqrtW, Phi1, Phi2, Phi3, SqrtWZ, Prop, n, p2, p3, n1, n2);

}

Betaprevprox = Beta;
Beta = Prop;
wsqlossBeta = wsqlossProp;
objprox(k) = wsqlossBeta + penalty(Gamma, Beta);
Iter(j) = k;

////proximal divergence check
if(objprox(k) > objprox(k - 1)){ascentprox = ascentprox + 1;}else{ascentprox = 0;}

////proximal convergence check
relobjprox = abs(objprox(k) - objprox(k - 1)) / (reltolprox + abs(objprox(k - 1))); 

if(k < maxiterprox && relobjprox < reltolprox){

df(j) = p - accu((Beta == 0));
Betas.col(j) = vectorise(Beta);
objprox.fill(NA_REAL);
break;

}else if(k == maxiterprox){

df(j) = p - accu((Beta == 0));
Betas.col(j) = vectorise(Beta);
objprox.fill(NA_REAL);
break;

}

}

////break proximal loop if maximum number of proximal backtraking step is reached
if(STOPprox == 1){break;}

}//end proximal loop

//stop program if maximum number of backtracking steps or maxiter is reached
if(STOPprox == 1){
  
endmodelno = j;
break;
  
}
  
}//end lambda loop

btenterprox = accu((BTprox > -1));
btiterprox = accu((BTprox > 0) % BTprox);

output = Rcpp::List::create(Rcpp::Named("Beta") = Betas,
                            Rcpp::Named("btenterprox") = btenterprox,
                            Rcpp::Named("btiterprox") = btiterprox,
                            Rcpp::Named("df") = df,
                            Rcpp::Named("endmodelno") = endmodelno,
                            Rcpp::Named("Iter") = Iter,
                            Rcpp::Named("lambda") = lambda,
                            Rcpp::Named("STOPmaxiter") = STOPmaxiter,
                            Rcpp::Named("STOPnewt") = STOPnewt,
                            Rcpp::Named("STOPprox") = STOPprox  
                            );  
  
}

}else{//general#################################################

if (iwls == "kron"){ // solve pure ls problems

////declare variables
int btnewtmax = 100;

double alphamax, alphanewt = 0.2, 
       delta, 
       L, loglikeBeta, logliketmp,  
       relobjnewt, relobjprox, 
       sqlossBeta, sqlossProp, snewt = 0.5, 
       tnewt, 
       valnewt;
            
arma::vec mwtrue(n3), 
          df(nlambda),
          eig1, eig2, eig3,
          objnewt(maxiternewt + 1), objprox(maxiterprox + 1);
            
arma::mat Beta(p1, p2 * p3),  Betaprevnewt(p1, p2 * p3), Betaprevprox(p1, p2 * p3), Betas(p, nlambda), BTnewt(nlambda, maxiternewt + 1), 
          DeltaBeta(p1, p2 * p3), 
          Eta(n1, n2 * n3), Etatmp(n1, n2 * n3), 
          Gamma(p1, p2 * p3), GradloglikeBeta(p1, p2 * p3), GradsqlossX(p1, p2 * p3), 
          Iter(nlambda, maxiternewt), 
          MuEta(n1, n2 * n3), MuEtatmp(n1, n2 * n3), 
          Phi1tW1Phi1, Phi2tW2Phi2, Phi3tW3Phi3,  PhitWZ, Prop(p1, p2 * p3), 
          SqrtW, SqrtW1(n1, n1), SqrtW2(n2, n2), SqrtW3(n3, n3), SqrtW1Phi1, SqrtW2Phi2, SqrtW3Phi3, SqrtWZ, Submat(n1, n2),
          W(n1, n2 * n3), W1(n1, n1), W2(n2, n2), W3(n3, n3), Wtrue(n1, n2 * n3), 
          X(p1, p2 * p3), 
          Z(n1, n2 * n3);

////fill variables
W3.fill(0);
SqrtW3.fill(0);

Iter.fill(0);
BTnewt.fill(-1);
Betas.fill(NA_REAL);
objnewt.fill(NA_REAL);
objprox.fill(NA_REAL);

////initialize
Betaprevprox.fill(0); //initialize at zero
Beta = Betaprevprox; 
Betaprevnewt = Beta;
Eta =  RHmat(Phi3, RHmat(Phi2, RHmat(Phi1, Beta, p2, p3), p3, n1), n1, n2);
loglikeBeta = loglike(Y, Weights, Eta, n, family);
MuEta = mu(Eta, family);

////lambda sequence
if(makelamb == 1){    
  
double lambdamax = as_scalar(max(max(abs(gradloglike(Y, Weights, Phi1, Phi2, Phi3, MuEta, Eta, 
                                                     n2, n3, p1, p2, n, family)) / penaltyfactor)));
double m = log(lambdaminratio);
double M = 0;
double difflamb = abs(M - m) / (nlambda - 1);
double l = 0;

for(int i = 0; i < nlambda ; i++){
  
lambda(i) = lambdamax * exp(l);
l = l - difflamb;

}

}else{std::sort(lambda.begin(), lambda.end(), std::greater<int>());}
                        
/////lambda loop
for (int j = 0; j < nlambda; j++){

Gamma = penaltyfactor * lambda(j);
objnewt(0) = loglikeBeta + penalty(Gamma, Betaprevprox);

/////kron iwls loop based on a kron newton approximation using kronecker structured weights.
for (int i = 0; i < maxiternewt; i++){

////true iwls weights W
Wtrue =  Weights % dmu(Eta, family) % dtheta(Eta, family);     //a * (mu’)^2 * theta’/mu’ = theta'mu'
////working responses (only relie on current Beta!)
Z = (Y - MuEta) % dg(MuEta, family) + Eta;

////tensor approximation of W, which dimension should we use??
for (int r = 0; r < n3; r++){

mwtrue(r) = mean(mean(Wtrue.cols(r * n2, (r + 1) * n2 - 1)));
Submat.fill(mwtrue(r));
W.cols(r * n2, (r + 1) * n2 - 1) = Submat;

}

////precompute
SqrtW = sqrt(W);
W3.diag() = mwtrue;
SqrtW3 = sqrt(W3);
SqrtW1Phi1 = Phi1;
SqrtW2Phi2 = Phi2;
SqrtW3Phi3 = SqrtW3 * Phi3;

Phi1tW1Phi1 = Phi1.t() * Phi1;
Phi2tW2Phi2 = Phi2.t() * Phi2;
Phi3tW3Phi3 = Phi3.t() * W3 * Phi3;

//proximal step size
eig1 = arma::eig_sym(Phi1tW1Phi1);
eig2 = arma::eig_sym(Phi2tW2Phi2);
eig3 = arma::eig_sym(Phi3tW3Phi3);

alphamax =  as_scalar(max(kron(eig1, kron(eig2 , eig3)))); 
L = alphamax / n;
delta = 1 / L; //can go up to 2 / L!

////precompute
SqrtWZ = SqrtW % Z;
PhitWZ = RHmat(Phi3.t(), RHmat(Phi2.t(), RHmat(Phi1.t(), W % Z, n2, n3), n3, p1), p1, p2);

/////proximal loop
for (int k = 0; k < maxiterprox; k++){

if(k == 0){
  
Betaprevprox = Beta;
objprox(0) = sqloss(SqrtW1Phi1, SqrtW2Phi2, SqrtW3Phi3, SqrtWZ, Beta, n, p2, p3, n1, n2) + penalty(Gamma, Beta);

}else{

X = Beta + (k - 2) / (k + 1) * (Beta - Betaprevprox);
GradsqlossX = (RHmat(Phi3tW3Phi3, RHmat(Phi2tW2Phi2, RHmat(Phi1tW1Phi1, X, p2, p3), p3, p1), p1, p2) - PhitWZ) / n;
Prop = prox_l1(X - delta * GradsqlossX, delta * Gamma);
sqlossProp = sqloss(SqrtW1Phi1, SqrtW2Phi2, SqrtW3Phi3, SqrtWZ, Prop, n, p2, p3, n1, n2);
Betaprevprox = Beta;
Beta = Prop;
sqlossBeta = sqlossProp;

Iter(j, i) = k;
objprox(k) = sqlossBeta + penalty(Gamma, Beta);
relobjprox = abs(objprox(k) - objprox(k - 1)) / abs(objprox(k - 1)); 

if(k < maxiterprox && relobjprox < reltolprox){
  
objprox.fill(NA_REAL);
break;

}else if(k == maxiterprox){

objprox.fill(NA_REAL);
break;

}

}

} //end proximal loop
 
/////newton line search
Eta =  RHmat(Phi3, RHmat(Phi2, RHmat(Phi1, Beta, p2, p3), p3, n1), n1, n2);
loglikeBeta = loglike(Y, Weights, Eta, n, family);
MuEta = mu(Eta, family);
GradloglikeBeta = gradloglike(Y, Weights, Phi1, Phi2, Phi3, MuEta, Eta, n2, n3, p1, p2, n, family);
DeltaBeta = Beta - Betaprevnewt;
valnewt = accu(GradloglikeBeta % DeltaBeta);
tnewt = 1;
BTnewt(j, i) = 0;

while (BTnewt(j, i) < btnewtmax) {

DeltaBeta = tnewt * DeltaBeta;
Etatmp =  RHmat(Phi3, RHmat(Phi2, RHmat(Phi1, Beta + DeltaBeta, p2, p3), p3, n1), n1, n2);
MuEtatmp = mu(Etatmp, family);
logliketmp = loglike(Y, Weights, Etatmp, n, family);

if(logliketmp <= loglikeBeta + alphanewt * tnewt * valnewt){

Beta = (1 - tnewt) * Betaprevnewt + tnewt * Beta;
break;

}else{

tnewt = snewt * tnewt;
BTnewt(j, i) = BTnewt(j, i) + 1;

}

}

if(tnewt < 1){//Beta has changed

Eta = RHmat(Phi3, RHmat(Phi2, RHmat(Phi1, Beta, p2, p3), p3, n1), n1, n2);
MuEta = mu(Eta, family);
loglikeBeta = loglike(Y, Weights, Eta, n, family);
objnewt(i + 1) = loglikeBeta + penalty(Gamma, Beta);  

}else{objnewt(i + 1) = loglikeBeta + penalty(Gamma, Beta);}

relobjnewt = abs(objnewt(i + 1) - objnewt(i)) / (reltolnewt + abs(objnewt(i)));
Betaprevnewt = Beta;

/////newton convergence check
if(relobjnewt < reltolnewt){//go to next lambda

df(j) = p - accu((Beta == 0));
Betas.col(j) = vectorise(Beta);
objnewt.fill(NA_REAL);
break;

}else if(i + 1 == maxiternewt){//go to next lambda

df(j) = p - accu((Beta == 0));
Betas.col(j) = vectorise(Beta);
objnewt.fill(NA_REAL);
break;

}

////check if maximum number of newton backtraking step is reached
if(BTnewt(j, i) >= btnewtmax){STOPnewt = 1;}

//check if maximum number of iterations for current lambda is reached
if(accu(Iter.row(j)) > maxiter){STOPmaxiter = 1;}

//break newton loop if maxiter or btnewtmax is reached
if(STOPmaxiter == 1 || STOPnewt == 1){break;}

} //end newton loop

//stop program if maximum number of backtracking steps or maxiter is reached
if(STOPmaxiter == 1 || STOPnewt == 1){

endmodelno = j;
break;

}

} //end lambda loop
            
btiternewt = accu((BTnewt > 0) % BTnewt);
            
output = Rcpp::List::create(Rcpp::Named("Beta") = Betas,
                            Rcpp::Named("btiternewt") = btiternewt,
                            Rcpp::Named("btiterprox") = btiterprox,
                            Rcpp::Named("endmodelno") = endmodelno,
                            Rcpp::Named("Iter") = Iter,
                            Rcpp::Named("lambda") = lambda,
                            Rcpp::Named("STOPmaxiter") = STOPmaxiter,
                            Rcpp::Named("STOPnewt") = STOPnewt,
                            Rcpp::Named("STOPprox") = STOPprox
                            );

}else if(iwls == "exact"){//solve wls suproblems

////declare variables
int ascentprox, ascentproxmax,
    btnewtmax, btprox;
    
double alphamax, alphanewt,
       delta, deltamin,
       Lmax, loglikeBeta, logliketmp, 
       relobjnewt, relobjprox, 
       snewt, sprox,
       tnewt, tprox, 
       valnewt, valprox, 
       wmax, wsqlossBeta, wsqlossProp, wsqlossX;  
       
arma::vec df(nlambda),
          eig1, eig2, eig3, 
          objnewt(maxiternewt + 1), objprox(maxiterprox + 1);

arma::mat Beta(p1, p2 * p3),  Betaprevnewt(p1, p2 * p3), Betaprevprox(p1, p2 * p3), Betas(p, nlambda), BTnewt(nlambda, maxiternewt + 1), 
          DeltaBeta(p1, p2 * p3),  
          Eta(n1, n2 * n3), Etatmp(n1, n2 * n3),
          Gamma(p1, p2 * p3), GradloglikeBeta(p1, p2 * p3), GradwsqlossX(p1, p2 * p3), 
          Iter(nlambda, maxiternewt),
          MuEta(n1, n2 * n3), MuEtatmp(n1, n2 * n3),
          Phi1tPhi1, Phi2tPhi2, Phi3tPhi3, PhitWZ, Prop(p1, p2 * p3), 
          SqrtW, SqrtWZ, 
          W(n1, n2 * n3), 
          X(p1, p2 * p3), 
          Z(n1, n2 * n3);
          
arma::cube BTprox(nlambda, maxiternewt, maxiterprox + 1);

////fill variables
ascentproxmax = 4;
btnewtmax = 100;
    
alphanewt = 0.2;
snewt = 0.5; 
sprox= 0.9;       
       
objnewt.fill(NA_REAL);
objprox.fill(NA_REAL);

Betas.fill(NA_REAL);
BTnewt.fill(-1); 
Iter.fill(0);

BTprox.fill(-1); //negative vals?

////precompute
if(nu > 0){
Phi1tPhi1 = Phi1.t() * Phi1;
Phi2tPhi2 = Phi2.t() * Phi2;
Phi3tPhi3 = Phi3.t() * Phi3;
eig1 = arma::eig_sym(Phi1tPhi1);
eig2 = arma::eig_sym(Phi2tPhi2);
eig3 = arma::eig_sym(Phi3tPhi3);
alphamax = as_scalar(max(kron(eig1, kron(eig2, eig3))));
}

////initialize 
Betaprevprox.fill(0); //initialize at 0
Beta = Betaprevprox;
Betaprevnewt = Beta;
Eta =  RHmat(Phi3, RHmat(Phi2, RHmat(Phi1, Beta, p2, p3), p3, n1), n1, n2);
loglikeBeta = loglike(Y, Weights, Eta, n, family);
MuEta = mu(Eta, family);

////make lambda sequence
if(makelamb == 1){
  
double lambdamax = as_scalar(max(max(abs(gradloglike(Y, Weights, Phi1, Phi2, Phi3, MuEta, Eta, 
                                                     n2, n3, p1, p2, n, family)) / penaltyfactor)));
double m = log(lambdaminratio);
double M = 0;
double difflamb = abs(M - m) / (nlambda - 1);
double l = 0;

for(int i = 0; i < nlambda ; i++){
  
lambda(i) = lambdamax * exp(l);
l = l - difflamb;

}

}else{std::sort(lambda.begin(), lambda.end(), std::greater<int>());}
        
////lambda loop
for (int j = 0; j < nlambda; j++){
  
Gamma = penaltyfactor * lambda(j);
objnewt(0) = loglikeBeta + penalty(Gamma, Beta);

/////newton loop
for (int i = 0; i < maxiternewt; i++){

////iwls weights
W = Weights % dmu(Eta, family) % dtheta(Eta, family); //a * (mu’)^2 * theta’/mu’
////working responses
Z = (Y - MuEta) % dg(MuEta, family) + Eta;

////precompute
SqrtW = sqrt(W);
SqrtWZ = SqrtW % Z;
PhitWZ = RHmat(Phi3.t(), RHmat(Phi2.t(), RHmat(Phi1.t(), W % Z, n2, n3), n3, p1), p1, p2);

////proximal step size
wmax = max(max(W));
Lmax = alphamax * wmax / n; //upper bound on Lipschitz constant
deltamin = 1.99 / Lmax; //minimum stepsize

//initial step size
if(nu > 0){delta = 1.9 / (nu * Lmax);}else{delta = 1;}

ascentprox = 0;
            
/////proximal loop
for (int k = 0; k < maxiterprox; k++){
  
if(k == 0){

Betaprevprox = Beta;
objprox(0) = wsqloss(SqrtW, Phi1, Phi2, Phi3, SqrtWZ, Beta, n, p2, p3, n1, n2) + penalty(Gamma, Beta);
if(nu > 0 & nu < 1){BTprox(j, i, k ) = 1;} //force initial backtracking for deltamin < delta

}else{

X = Beta + (k - 2) / (k + 1) * (Beta - Betaprevprox);
GradwsqlossX = (winprod(W, Phi1, Phi2, Phi3, X, n1, n2, n3, p1, p2, p3) - PhitWZ) / n;

////check if proximal backtracking occurred last iteration
if(BTprox(j, i, k - 1) > 0){btprox = 1;}else{btprox = 0;}

////check for divergence
if(ascentprox > ascentproxmax){btprox = 1;}

if((btprox == 1 & deltamin < delta) || nu == 0){//backtrack
                        
wsqlossX = wsqloss(SqrtW, Phi1, Phi2, Phi3, SqrtWZ, X, n, p2, p3, n1, n2);

////proximal line search
BTprox(j, i, k) = 0;

while (BTprox(j, i, k) < btproxmax){

Prop = prox_l1(X - delta * GradwsqlossX, delta * Gamma);
wsqlossProp = wsqloss(SqrtW, Phi1, Phi2, Phi3, SqrtWZ, Prop, n, p2, p3, n1, n2);
valprox = as_scalar(wsqlossX + accu(GradwsqlossX % (Prop - X)) + 1 / (2 * delta) * sum_square(Prop - X));

if (wsqlossProp <= valprox + 0.0000001){ //need to add a little due to numerical issues

break;

}else{

delta = sprox * delta;
BTprox(j, i, k) = BTprox(j, i, k) + 1;

if(delta < deltamin){delta = deltamin;}

}

}

////check if maximum number of proximal backtraking step is reached
if(BTprox(j, i, k) == btproxmax){STOPprox = 1;}

}else{//no backtracking

Prop = prox_l1(X - delta * GradwsqlossX, delta * Gamma);
wsqlossProp = wsqloss(SqrtW, Phi1, Phi2, Phi3, SqrtWZ, Prop, n, p2, p3, n1, n2);

}

Betaprevprox = Beta;
Beta = Prop;
wsqlossBeta = wsqlossProp;
objprox(k) = wsqlossBeta + penalty(Gamma, Beta);

////check if objective has increased
if(objprox(k) > objprox(k - 1)){ascentprox = ascentprox + 1;}else{ascentprox = 0;}

Iter(j, i) = k;

////proximal convergence check
relobjprox = abs(objprox(k) - objprox(k - 1)) / (reltolprox + abs(objprox(k - 1))); 

if(k < maxiterprox && relobjprox < reltolprox){

objprox.fill(NA_REAL);
break;

}else if(k == maxiterprox){

objprox.fill(NA_REAL);
break;

}

}

////break proximal loop if maximum number of proximal backtraking step is reached
if(STOPprox == 1){break;}

////break proximal loop if maximum number of iterations is reached
if(accu(Iter.row(j)) > maxiter){
  
STOPmaxiter = 1;
break;
  
}

} //end proximal loop
            
/////newton line search
Eta =  RHmat(Phi3, RHmat(Phi2, RHmat(Phi1, Beta, p2, p3), p3, n1), n1, n2);
loglikeBeta = loglike(Y, Weights, Eta, n, family);
MuEta = mu(Eta, family);
GradloglikeBeta = gradloglike(Y, Weights, Phi1, Phi2, Phi3, MuEta, Eta, n2, n3, p1, p2, n, family);
DeltaBeta = Beta - Betaprevnewt;
valnewt = accu(GradloglikeBeta % DeltaBeta);
tnewt = 1;
BTnewt(j, i) = 0;

while (BTnewt(j, i) < btnewtmax) {

DeltaBeta = tnewt * DeltaBeta;
Etatmp = RHmat(Phi3, RHmat(Phi2, RHmat(Phi1, Beta + DeltaBeta, p2, p3), p3, n1), n1, n2);
MuEtatmp = mu(Etatmp, family);
logliketmp = loglike(Y, Weights, Etatmp, n, family);

if(logliketmp <= loglikeBeta + alphanewt * tnewt * valnewt){

Beta = (1 - tnewt) * Betaprevnewt + tnewt * Beta;
break;

}else{

tnewt = snewt * tnewt;
BTnewt(j, i) = BTnewt(j, i) + 1;

}

}

if(tnewt < 1){//Beta has changed

Eta = RHmat(Phi3, RHmat(Phi2, RHmat(Phi1, Beta, p2, p3), p3, n1), n1, n2);
MuEta = mu(Eta, family);
loglikeBeta = loglike(Y, Weights, Eta, n, family);
objnewt(i + 1) = loglikeBeta + penalty(Gamma, Beta);  

}else{objnewt(i + 1) = loglikeBeta + penalty(Gamma, Beta);}

relobjnewt = abs(objnewt(i + 1) - objnewt(i)) / (reltolnewt + abs(objnewt(i)));
Betaprevnewt = Beta;

/////newton convergence check
if(relobjnewt < reltolnewt){//go to next lambda

df(j) = p - accu((Beta == 0));
Betas.col(j) = vectorise(Beta);
objnewt.fill(NA_REAL);
break;

}else if(i + 1 == maxiternewt){//go to next lambda

df(j) = p - accu((Beta == 0));
Betas.col(j) = vectorise(Beta);
objnewt.fill(NA_REAL);
break;

}

////check if maximum number of newton backtraking step is reached
if(BTnewt(j, i) >= btnewtmax){STOPnewt = 1;}

////break newton loop if maximum number of backtracking steps or maxiter is reached 
if(STOPprox == 1 || STOPmaxiter == 1 || STOPnewt == 1){break;}

} //end newton loop

//stop program if maximum number of backtracking steps or maxiter is reached
if(STOPnewt == 1 || STOPprox == 1 || STOPmaxiter == 1){
  
endmodelno = j;
break;
  
}
  
} //end lambda loop

btiternewt = accu((BTnewt > 0) % BTnewt);
btenterprox = accu((BTprox > -1));
btiterprox = accu((BTprox > 0) % BTprox);

output = Rcpp::List::create(Rcpp::Named("Beta") = Betas,
                            Rcpp::Named("btenterprox") = btenterprox,
                            Rcpp::Named("btiternewt") = btiternewt,
                            Rcpp::Named("btiterprox") = btiterprox,
                            Rcpp::Named("df") = df,
                            Rcpp::Named("endmodelno") = endmodelno,
                            Rcpp::Named("Iter") = Iter,
                            Rcpp::Named("lambda") = lambda,
                            Rcpp::Named("STOPmaxiter") = STOPmaxiter,
                            Rcpp::Named("STOPnewt") = STOPnewt,
                            Rcpp::Named("STOPprox") = STOPprox
                            );  

}

}

return output;

} //end function

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// objective values  ///////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//[[Rcpp::export]]
Rcpp::List getobj(arma::mat Y, arma::mat Weights,
                  arma::mat Phi1, arma::mat Phi2, arma::mat Phi3,
                  Rcpp::NumericVector beta,
                  arma::vec lambda,
                  arma::mat penaltyfactor,
                  std::string family){

Rcpp::NumericVector vecbeta(beta);
Rcpp::IntegerVector BetaDim = vecbeta.attr("dim");
arma::cube Beta(vecbeta.begin(), BetaDim[0], BetaDim[1], BetaDim[2], false);

int p1 = Phi1.n_cols;
int p2 = Phi2.n_cols;
int p3 = Phi3.n_cols;
int p = p1 * p2 * p3;
int n1 = Phi1.n_rows;
int n2 = Phi2.n_rows;
int n3 = Phi3.n_rows;
int n = n1 * n2 * n3;    
int nlambda = lambda.n_elem;

arma::mat Eta, MuEta;

arma::vec Obj(nlambda), Loss(nlambda), Pen(nlambda);

for (int j = 0; j < nlambda; j++){
  
Pen(j) = penalty(penaltyfactor * lambda(j), Beta.slice(j));
Eta = RHmat(Phi3, RHmat(Phi2, RHmat(Phi1, Beta.slice(j), p2, p3), p3, n1), n1, n2);
MuEta = mu(Eta, family);
Loss(j) = loglike(Y, Weights, Eta, n, family);
Obj(j) = Loss(j) + Pen(j);

}

Rcpp::List output = Rcpp::List::create(Rcpp::Named("Obj") = Obj,
                                       Rcpp::Named("Loss") = Loss,
                                       Rcpp::Named("Pen") = Pen);
return output;

}