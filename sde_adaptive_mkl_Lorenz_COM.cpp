/*
 * sde_adaptive_mkl_Lorenz_COM.cpp
 *
 *  Created on: 10 Mar 2017
 *      Author: fangw
 */



#include <iostream>
#include <algorithm>      // std::max
#include <cmath>
#include "mlmcwei.hpp"
#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <mkl_vsl.h>
#include <memory.h>
#include <omp.h>



// macros for rounding x up or down to a multiple of y
#define ROUND_UP(x, y) ( ( ((x) + (y) - 1) / (y) ) * (y) )
#define ROUND_DOWN(x, y) ( ((x) / (y)) * (y) )

// each OpenMP thread has its own VSL RNG and storage
#define RV_NUM 65536
double           *dW;
VSLStreamStatePtr stream;
#pragma omp threadprivate(stream, dW)

// dimension of the SDE
#define D 6


void sde_adaptive_l(int, int, double *);
void pathcalc(double,double *,int,int,double *,double *,double *,double *,double *,double *,double *,double*);
void drift(double* Xf, double* driftf);
double dt(double* X, double* Y, int l);
double P(double* X);
double Spring(double *X);


int main(int argc, char **argv) {

  int M  = 0;     // refinement cost factor
  int N0 = 1000;   // initial samples on each level
  int Lmin = 2;   // minimum refinement level
  int Lmax = 10;  // maximum refinement level

  int N, L;
  char filename[32];
  FILE *fp;

  sprintf(filename,"sde_adaptive_lorenz_COM_Sb.txt");
  fp = fopen(filename,"w");


  N      = 50000;    // samples for convergence tests
  L      = 5;        // levels for convergence tests
//  float Eps[] = { 0.0001, 0.0002, 0.0005, 0.001, 0.002 };
  float Eps[] = { 0.1, 0.05, 0.02, 0.01, 0.005, 0.0};
//  float Eps[] = { 1, 0.5, 0.05, 0.01, 0.0};
  mlmc_test(sde_adaptive_l, M, N, L, N0, Eps, Lmin,Lmax, fp);

  fclose(fp);
  return 0;
}

/*-------------------------------------------------------
%
% level l estimator
%
*/


void sde_adaptive_l(int l, int N, double *sums) {

	double T=10.0, X0[D], sum1=0.0, sum2=0.0, sum3=0.0, sum4=0.0, sum5=0.0, sum6=0.0, sum7=0.0,sum8=0.0;
  X0[0] = -2.4;
  X0[1] = -3.7;
  X0[2] = 14.98;
  X0[3] = 0.0;
  X0[4] = 0.0;
  X0[5] = 0.0;

#pragma omp parallel shared(T,X0,N,l)	\
                     reduction(+:sum1,sum2, sum3, sum4, sum5, sum6, sum7,sum8)
  {
    double sum1_t=0.0, sum2_t=0.0, sum3_t=0.0, sum4_t=0.0, sum5_t=0.0, sum6_t=0.0, sum7_t=0.0,sum8_t=0.0;
    int    num_t  = omp_get_num_threads();
    int    tid    = omp_get_thread_num();

 //   printf("tid=%d, creating RNG generator and allocating memory \n",tid);

    // create RNG, then give each thread a unique skipahead
    vslNewStream(&stream, VSL_BRNG_MRG32K3A,1337);
    long long skip = ((long long) (tid+1)) << 48;
    vslSkipAheadStream(stream,skip);

    dW = (double *)malloc(RV_NUM*sizeof(double)); //each double needs 8 bytes

    int N3 = ROUND_UP(((tid+1)*N)/num_t,1)
           - ROUND_UP(( tid   *N)/num_t,1);
 //   printf("N3=%d \n",N3);

    pathcalc(T,X0,N3,l, &sum1_t, &sum2_t, &sum3_t,&sum4_t, &sum5_t, &sum6_t,&sum7_t,&sum8_t);
    sum1 += sum1_t;
    sum2 += sum2_t;
    sum3 += sum3_t;
    sum4 += sum4_t;
    sum5 += sum5_t;
    sum6 += sum6_t;
    sum7 += sum7_t;
    sum8 += sum8_t;


  }

  for (int k=0; k<8; k++) sums[k] = 0.0;
  sums[0]=sum1;
  sums[1]=sum2;
  sums[2]=sum3;
  sums[3]=sum4;
  sums[4]=sum5;
  sums[5]=sum6;
  sums[6]=sum7;
  sums[7]=sum8;


  // delete generator and storage
  #pragma omp parallel
    {
      vslDeleteStream(&stream);
      free(dW);
    }
}


void pathcalc(double T,double * X0,int N,int l,double *sum1_t,double *sum2_t, double *sum3_t,double *sum4_t,
		double *sum5_t, double *sum6_t, double *sum7_t,double *sum8_t){

      double sum1=0.0, sum2=0.0, sum3=0.0, sum4=0.0, sum5=0.0, sum6=0.0,sum7=0.0,sum8=0.0;
      double sigmaf=6.0,sigmac=6.0, S=10.0;
    // sigmaf = 1;
    // sigmac = 1;

	  // work out max number of paths in a group
      int M = std::pow(2,l+11)*4*32;
          M = std::min(RV_NUM,M);
	    int N2 = std::max(ROUND_DOWN(RV_NUM/M,1),1);
      if((M*N2)!=RV_NUM)
        std::cout<<"dashabi"<<M<<N2<<"\n";

	  // loop over all paths in groups of size N2
	    for (int n0=0; n0<N; n0+=N2) {
	      // may have to reduce size of final group
	      if (N2>N-n0) N2 = N-n0;
	      int ns =0;
	      // generate required random numbers for this group
	      vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,
	                    stream,RV_NUM,dW,0,1);
	      // loop over paths within group in increments of VECTOR_LENGTH
	      for (int n1=0; n1<N2; n1+=1) {
	      //    int offset = n1*M; // number of random numbers already used
	          double Xf[D],Xc[D],dWf[D],dWc[D],ddW[D],driftf[D],driftc[D];
	          double Xcold[D], Xfold[D], lamdaf[D], lamdac[D], Cmf[D],Cmc[D], Springf,Springc;  //ss
	      	  for(int i=0; i<D; i++){
	      		     Xf[i]=X0[i];Xc[i]=X0[i];
                 dWf[i]=0.0;dWc[i]=0.0;ddW[i]=0.0;
	      		     Xcold[i]=X0[i];Xfold[i]=X0[i]; //ss
	      		     lamdac[i]=0.0; lamdaf[i]=0.0;
                 driftf[i]=0.0; driftc[i]=0.0;
                 Cmc[i]=0.0; Cmf[i]=0.0;
	          }
	          Springf=0.0; Springc=0.0;
	      	  drift(Xf,driftf);
	      	  drift(Xc,driftc);

	          double t=0.0,tf=0.0,tc=0.0,tm = 0.0;
	          double hf=0.0,hc= 0.0;
	          double Pf=0.0,Pc=0.0,dP=0.0,ss=5.0;
	       //   int ns = 0;

	          if (l==0) {
	              	while(t<T){
                        if(ns>(RV_NUM-2*D-1)){
		    	  		               vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,
		    	  	                    stream,RV_NUM,dW,0,1);
		    	  		        sum1+=ns;
		    	  		        ns = 0;
		      		          }
	              		    t = tf;

	              		    for(int i=0; i<3; i++){
	                         ddW[i] = std::sqrt(hf)*dW[ns];
                           Pf += Xf[3+i]*ddW[i];
	                         ns +=1;
	              		    }


	                      for(int i=0; i<D; i++){
	                    	   Xf[i] += driftf[i]*hf + sigmaf*ddW[i];
	                      }

                        drift(Xf,driftf);
	                      hf = std::min(dt(Xf,driftf,l), T-tf);
	                      tf += hf;

	                }
	           }
	           else {
	              	while(t<T){
                        if(ns>(RV_NUM-2*D-1)){
		    	  		               vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,
		    	  	                       stream,RV_NUM,dW,0,1);
		    	  		               sum1+=ns;
		    	  		               ns = 0;
                        }
	              		    tm=t;
	              		    t = std::min(tf,tc);

	              		    for(int i=0; i<3; i++){
	              			        while(std::isnan(dW[ns])){
	              				             std::cout<<"dW you wen ti!!\n";
	              				             ns +=1;
	              			        }
	              			        ddW[i] = std::sqrt(t-tm)*dW[ns];

	              			        dWf[i] += ddW[i];
	              			        dWc[i] += ddW[i];
	              			        ns +=1;
	              		    }

	                      if(tf==t){
                            for(int i=0; i<3; i++){
                                Pf += Xf[i+3]*(dWf[i]+lamdaf[i]/sigmaf*hf);
                            }
	                    	    for(int i=0; i<D; i++){
		                    	      Xf[i]  += driftf[i]*hf + sigmaf*dWf[i];
		                            Cmf[i]  += -lamdaf[i]*dWf[i]/sigmaf-0.5*lamdaf[i]*lamdaf[i]*hf/sigmaf/sigmaf;
		                            Xcold[i] = Xc[i] + driftc[i]*(tf-tc+hc) + sigmac*dWc[i];
	                    	    }
	                    	    //Spring(Xf,Springf);
                            Springf = Spring(Xf);

				                    for(int i=0; i<3; i++){
		                            lamdaf[i]= Springf*(Xcold[i]-Xf[i]);
		                        }
                            drift(Xf,driftf);
	                          hf = std::min(dt(Xf,driftf,l),T-tf);
	                          tf += hf;
	                          for(int i=0;i<D;i++){
	                        	    dWf[i] = 0;
	                              driftf[i] += lamdaf[i];
	                          }
	                      }
	                      if(tc==t){
                            for(int i=0; i<3; i++){
                                Pc += Xc[i+3]*(dWc[i]+lamdac[i]/sigmac*hc);
                            }
	                    	    for(int i=0; i<D; i++){
	                    		      Xc[i] += driftc[i]*hc + sigmac*dWc[i];
	                    	        Cmc[i] += -lamdac[i]*dWc[i]/sigmac-0.5*lamdac[i]*lamdac[i]*hc/sigmac/sigmac;
	                    	        Xfold[i]= Xf[i] + driftf[i]*(tc-tf+hf) + sigmaf*dWf[i];
	                    	    }
	                          Springc = Spring(Xc);//Spring(Xc,Springc);
	                    	    for(int i=0; i<3; i++){
	                    	        lamdac[i]= Springc*(Xfold[i]-Xc[i]);
	                    	    }
	                          //	if((lamdac[0]+lamdac[1]+lamdac[2])!=0)
	                          //		std::cout<<lamdac[0]<<" "<<lamdac[1]<<" "<<lamdac[2]<<"\n";
	                    	    drift(Xc,driftc);
	                    	    hc = std::min(dt(Xc,driftc,l-1),T-tc);
	                    	    tc += hc;
	                    	    for(int i=0;i<D;i++){
	                    		      dWc[i] = 0;
	                    		      driftc[i] += lamdac[i];
	                    	     }
                        }
	              	}
	             }

	            Pf = (Pf*S/sigmaf*Xf[2]+Xf[5])*exp(Cmf[0]+Cmf[1]+Cmf[2]);
              //Pf = Xf[5]*exp(Cmf[0]+Cmf[1]+Cmf[2]);
	             // std::cout<<Pf<<"\n";
	             if(l>0){
	                 Pc = (Pc*S/sigmac*Xc[2]+Xc[5])*exp(Cmc[0]+Cmc[1]+Cmc[2]);
                   //Pc = Xc[5]*exp(Cmc[0]+Cmc[1]+Cmc[2]);
                   dP = Pf-Pc;
	             }
	             else{
	                  Pc = 0;
	                  dP = Pf;
	             }

               if(std::isnan(Pf) || std::isnan(Pc)){
	                //  	 std::cout<<"da ben dan\n";
	            	    Pf=0;
	            	    Pc=0;
	            	    dP=0;
	            	    n1-=1;
	             }

	          sum2 += dP;
	          sum3 += dP*dP;
	          sum4 += dP*dP*dP;
	          sum5 += dP*dP*dP*dP;
	          sum6 += Pf;
	          sum7 += Pf*Pf;
	          sum8 += abs(Xf[0]-Xc[0])>1;

	       }
	      sum1 += ns;
	      if(ns<0.5*M*N2){
	    	//  std::cout<<"the number of RVs left is too many.\n";
	      }

	    }
	    *sum1_t = sum1;
	    *sum2_t = sum2;
	    *sum3_t = sum3;
	    *sum4_t = sum4;
	    *sum5_t = sum5;
	    *sum6_t = sum6;
	    *sum7_t = sum7;
	    *sum8_t = sum8;


}


void drift(double *X, double *driftf)
{
	double sig=10.0,beta=8.0/3.0,rho=28.0;
  double S = 10;
	driftf[0] = sig*(X[1]-X[0]);
	driftf[1] = X[0]*(rho-X[2])-X[1];
	driftf[2] = X[0]*X[1]-beta*X[2];
  driftf[3] = -(sig+S)*X[3] +sig*X[4];
  driftf[4] = X[0]+ (rho-X[2])*X[3]- (1.0+S)*X[4] -X[0]*X[5];
  driftf[5] = X[1]*X[3]+ X[0]*X[4] - (beta+S)*X[5];
}
double dt(double *X, double *Y, int l)
{
  double a = X[0]*X[0]+X[1]*X[1]+X[2]*X[2];
	double b = Y[0]*Y[0]+Y[1]*Y[1]+Y[2]*Y[2];
  //return std::pow(0.5,l+8);
	return std::pow(0.5,l+7.0)*std::max(100.0,a)/std::max(100.0,b);
}


double P(double* X)
{
	double sum=0.0;
	for(int i=0;i<D;i++)
		sum+=X[i]*X[i];
	return std::sqrt(sum);
}


double Spring(double* X)
{
	/*double dach = 412216.0 +3969.0*X[0]*X[0] + 7290.0*X[0]*X[1]- 13770.0*X[2];
	double xach = -8179.0 + 27.0*X[0]*X[0] + 270.0*X[2];
	std::complex<double> z(dach*dach+4*xach*xach*xach,0),i(0.0,1.0),k(1.0/3.0,0.0),costt(-41.0/9.0,0.0),unit1(1.0,0.0);
	std::complex<double> cach(dach,0);
  cach = std::pow(cach+std::sqrt(z),k);

	std::complex<double> lamda1 =  costt + std::pow(2.0,1.0/3.0)*xach/9.0/cach-cach/9.0/std::pow(2.0,1.0/3.0);
	std::complex<double> lamda2 =  costt - (unit1+std::sqrt(3.0)*i)*xach/cach/9.0/std::pow(2.0,2.0/3.0) + (unit1-std::sqrt(3.0)*i)*cach/18.0/std::pow(2.0,1.0/3.0);
	std::complex<double> lamda3 =  costt - (unit1-std::sqrt(3.0)*i)*xach/cach/9.0/std::pow(2.0,2.0/3.0) + (unit1+std::sqrt(3.0)*i)*cach/18.0/std::pow(2.0,1.0/3.0);

	double Yy = std::max(real(lamda1),real(lamda2));
	Yy = std::max(real(lamda3),Yy);
	Yy = std::max(Yy,1.0);
	Yy = std::min(Yy,28.0);
	//Y[0]= Yy;
	//Y[1]= Yy;
	//Y[2]= Yy;*/
  double Yy= 0.0;
  return Yy;

		/*
	double sigma=10.0,beta=8.0/3.0,rho=28.0;
    Y[0] = 0.0;
    Y[1] = std::max(0.0,rho-1.0-X[0]-X[2]);
	Y[2] = std::max(0.0,X[1]+X[0]-beta);
    */

	//Y[0] =0.0;
	//Y[1] =0.0;
	//Y[2] =0.0;


}
