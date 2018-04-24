/*
 * mlmcwei.hpp
 *
 *  Created on: 4 Mar 2017
 *      Author: fangw
 */

#ifndef MLMCWEI_HPP_
#define MLMCWEI_HPP_


#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define PRINTF2(fp, ...) {printf(__VA_ARGS__);fprintf(fp,__VA_ARGS__);}

void regression(int, float *, float *, float &a, float &b);

float mlmc(int Lmin, int Lmax, int N0, float eps,
           void (*mlmc_l)(int, int, double *),
           float alpha_0,float beta_0,float gamma_0, int *Nl, float *Cl);



void mlmc_test(void (*mlmc_l)(int, int, double *), int M,int N,int L,
               int N0, float *Eps, int Lmin, int Lmax, FILE *fp);


#endif /* MLMCWEI_HPP_ */
