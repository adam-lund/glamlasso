/*  
    A number of functions utilized by rcppfunc.cpp. The functions fall into two categories
    auxiliary functions and model specific functions. 
    The auxiliary functions are independent of the specific GLAM. 
    The model specific functions are specific to the GLAM. 
    To add a new GLAM to the glamlasso package you need to specify:
    
    g - link function, 
    dg - derivative of g, 
    mu - inverse link function,
    dmu - derivative of mu,
    theta - canonical parameter function,  
    dtheta - derivative of theta,
    b - cummulant function.

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
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

//#define TIMING
//#include <RcppArmadillo.h>
//#include "/Users/adamlund/Documents/KU/Phd/Project/Computer/Vincent/timer/simple_timer.h"

#include <math.h>
//#include <cmath>
//#include <tr1/cmath>
using namespace std;
//using namespace arma;



////////////////////////////////// Auxiliary functions
//////////////////// Direct RH-transform of a flat 3d array (matrix) M by a matrix X
arma::mat RHmat(arma::mat const& X, arma::mat const& M,int col, int sli){
    
//    TIMER_START
    
int rowx = X.n_rows;
    
////matrix multiply
arma::mat XM = X * M;
    
////make matrix into rotated (!) cube (on matrix form)
arma::mat Mnew(col, sli * rowx);
for (int s = 0; s < sli; s++) {
  
for (int c = 0; c < col; c++) {
  
for (int r = 0; r < rowx; r++) {
  
Mnew(c, s + r * sli) = XM(r,c + s * col);

}

}

}

return Mnew;

}

//////////////////// weigthed product
arma::mat wprod(arma::mat const& W, 
                arma::mat const& Phi1, arma::mat const& Phi2, arma::mat const& Phi3, 
                arma::mat const& B, int n1, int n2, int p2, int p3){
    
//TIMER_START
    
arma::mat XB = RHmat(Phi3, RHmat(Phi2, RHmat(Phi1, B, p2, p3), p3, n1), n1, n2);

arma::mat WXB = W % XB;

return WXB;

}

////////////////// weigthed inner product
arma::mat winprod(arma::mat const& W, 
                  arma::mat const& Phi1, arma::mat const& Phi2, arma::mat const& Phi3, 
                  arma::mat const& B, int n1, int n2, int n3, int p1, int p2, int p3){
    
//TIMER_START
    
arma::mat XB = RHmat(Phi3, RHmat(Phi2, RHmat(Phi1, B, p2, p3), p3, n1), n1, n2);

arma::mat WXB = W % XB;

arma::mat XWXB = RHmat(Phi3.t(), RHmat(Phi2.t(), RHmat(Phi1.t(), WXB, n2, n3), n3, p1), p1, p2);

return XWXB;

}

//////////////////// Proximal operator (soft threshold)
arma::mat prox_l1(arma::mat const& zv, arma::mat const& gam){
    
//TIMER_START
    
return (zv >= gam) % (zv - gam) + (zv <= -gam) % (zv + gam);

}

//////////////////// Penalty function
double penalty(arma::mat const& gam,  arma::mat const& zv){return accu(gam % abs(zv));}

//////////////////// Sum of squares function
double sum_square(arma::mat const& x){return accu(x % x);}

//////////////////// Weighted square loss function
double wsqloss(arma::mat const& W, 
               arma::mat const& A1, arma::mat const& A2, arma::mat const& A3, 
               arma::mat const& b, arma::mat const& x, 
               int nv, int c2, int c3, int r1, int r2) {
    
//TIMER_START
    
arma::mat Ax = wprod(W, A1, A2, A3, x, r1, r2, c2, c3);

return 0.5 * sum_square(Ax - b) / nv;

}

//////////////////// Square loss function
double sqloss(arma::mat const& A1, arma::mat const& A2, arma::mat const& A3, 
              arma::mat const& y, arma::mat const& x, int nv, int c2, int c3, int r1, int r2) {
    
//TIMER_START
    
arma::mat Ax = RHmat(A3, RHmat(A2, RHmat(A1, x, c2, c3), c3, r1), r1, r2);

return 0.5 * sum_square(Ax - y) / nv;

}

////////////////////////Model specific functions

//////////////////// Link function g - as function of mean mu
arma::mat g(arma::mat const& x, string fam){
  
arma::mat out;
if(fam == "binomial"){

//out = log(mu / (Weights - mu)); //mean param
out = log(x / (1 - x)); //proportion param

}else if(fam == "poisson"){

out = log(x);

}else if(fam == "gaussian"){
  
out = x;

}else if(fam == "gamma"){

out = log(x);

}

return out;

}

arma::mat dg(arma::mat const& x, string fam){

arma::mat out;

if(fam == "binomial"){

//out = Weights / (x % (Weights - x)); //mean param
out = 1 / (x % (1 - x)); // / Weights; //proportion param

}else if(fam == "poisson"){

out = 1 / x;

}else if(fam == "gaussian"){
  
out = 0 * x + 1;

}else if(fam == "gamma"){

out = 1 / x;

}

return out;

}
    
//////////////////// Inverse link function or mean value function mu - as function of eta
arma::mat mu(arma::mat const& eta, string fam){

arma::mat out;

if(fam == "binomial"){  

arma::mat  expeta = exp(eta);
//out =  Weights % expeta / (expeta + 1); //mean param - the function returns mean mu = w * p 
out =  expeta / (expeta + 1);           //proportion param - the function returns proportion p = mu / w 

}else if(fam == "poisson"){

out = exp(eta);

}else if(fam == "gaussian"){
  
out = eta;
  
}else if(fam == "gamma"){

out = exp(eta);

}

return out;

}

arma::mat dmu(arma::mat const& eta, string fam){
  
arma::mat out;

if(fam == "binomial"){

arma::mat expeta = exp(eta);
//out = Weights % expeta / pow(expeta + 1, 2); //mean param
out = expeta / pow(expeta + 1, 2);           //proportion param 

}else if(fam == "poisson"){

out = exp(eta);

}else if(fam == "gaussian"){
  
out =  0 * eta + 1;

}else if(fam == "gamma"){

out = exp(eta);

}

return out;

}
    
//////////////////// theta - as function eta
arma::mat theta(arma::mat const& eta, string fam) {

arma::mat out;

if(fam == "binomial"){

out = eta; // with canonical link (logit)

}else if(fam == "poisson"){

out = eta; // with canonical link (log)

}else if(fam == "gaussian"){

out = eta;

}else if(fam == "gamma"){

out = -exp(-eta); // with non-canonical link (log)

}

return out;

}

arma::mat dtheta(arma::mat const& eta, string fam) {

arma::mat out;

if(fam == "binomial"){

out = 0 * eta + 1; // = 1 mat  with canonical link (logit)

}else if(fam == "poisson"){

out = 0 * eta + 1; // = 1 mat

}else if(fam == "gaussian"){
  
out = 0 * eta + 1; // = 1 mat

}else if(fam == "gamma"){

out = exp(-eta);

}

return out;

}

//////////////////// Cumulant function  b (aka kappa) - as function of theta
arma::mat b(arma::mat const& theta, string fam) {

arma::mat out;

if(fam == "binomial"){

//out = Weights % log(1 + exp(theta)); //mean param
out = log(1 + exp(theta));           //proportion param with no Weights

}else if(fam == "poisson"){

out = exp(theta);

}else if(fam == "gaussian"){

out = pow(theta, 2) / 2;

}else if(fam == "gamma"){

out = -log(-theta);
//  out = -1 / x;
//  out = pow(x, -2);

}

return out;

}
    
//////////////////// constant term  C, c or tau SHOULD IT INCLUDE PRIOR WEIGHTS A????
//arma::mat c(arma::mat const& y, string fam) {

//arma::mat out(y.n_rows, y.n_cols);

//if(fam == "binomial"){

//out = y * 0; //WRONG!!!!!!!!!!!!

//}else if(fam == "poisson"){

//for (int i = 0; i < out.n_rows; i++){
  
//for (int j = 0; j < out.n_cols; j++){

//out(i, j) = - std::lgamma(y(i, j) + 1);

//}

//}

//}else if(fam == "gaussian"){

//out = - pow(y, 2) / 2 - log(2 * 3.141592653589793238463) / 2;

//}else if(fam == "gamma"){

//out = y * 0; //WRONG!!!!!!!!!!!!

//}

//return out;

//}

////////////////////   negative log-likelihood-function
double loglike(arma::mat const& Y, 
               arma::mat const& Weights,   
               arma::mat const& eta, 
               int n,        
               string fam){          
                 
double out;
arma::mat thetaeta = theta(eta, fam);

arma::mat tmp = b(thetaeta, fam) - Y % thetaeta; //- c(Y, fam);

//out = accu(tmp) / n;
out = accu(Weights % tmp) / n;

return out;

}
    
//////////////////// Gradient of negative log-likelihood-function
arma::mat gradloglike(arma::mat const& Y, 
                      arma::mat const& Weights, 
                      arma::mat const& Phi1, arma::mat const& Phi2, arma::mat const& Phi3, 
                      arma::mat const& mueta, arma::mat const& eta,
                      int n2, int n3, int p1, int p2, int n, 
                      string fam){

arma::mat out;

arma::mat tmp = dtheta(eta, fam) % (mueta - Y);
    
out = RHmat(Phi3.t(), RHmat(Phi2.t(), RHmat(Phi1.t(), Weights % tmp, n2, n3), n3, p1), p1, p2) / n;

return out;

}
