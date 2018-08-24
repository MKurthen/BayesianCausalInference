/*  Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
*/


#include "mex.h"
#include "hsic.h"
#include <cmath>


/* Input Arguments */

#define X_IN       prhs[0]
#define Y_IN       prhs[1]
#define SX_IN      prhs[2]
#define SY_IN      prhs[3]
#define NRPERM_IN  prhs[4]
#define NR_IN      2
#define NR_IN_OPT  3


/* Output Arguments */

#define P_OUT        plhs[0]
#define HSIC_OUT     plhs[1]
#define PROB_OUT     plhs[2]
#define LOGP_OUT     plhs[3]
#define DHSICDY_OUT  plhs[4]
#define NR_OUT       0
#define NR_OUT_OPT   5


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] ) { 
    /* Check for proper number of arguments */
    if( ((nrhs < NR_IN) || (nrhs > NR_IN + NR_IN_OPT)) || ((nlhs < NR_OUT) || (nlhs > NR_OUT + NR_OUT_OPT)) ) { 
        mexErrMsgTxt("Usage: [p,HSIC,prob,logp,dHSICdY] = fasthsic(X,Y,[sX,sY,nrperm])\n\n"
        "\n"
        "Calculates the Hilbert-Schmidt Independence Criterion between X and Y using RBF kernels\n"
        "  kX(i,j) = exp(-norm(X(i,:) - X(j,:))^2 / (2 * sX^2)) and\n"
        "  kY(i,j) = exp(-norm(Y(i,:) - Y(j,:))^2 / (2 * sY^2))\n"
        "\n"
        "INPUT:    X       = Nxd1 vector of doubles\n"
        "          Y       = Nxd2 vector of doubles\n"
        "optional: sX      = kernel bandwidth for X (automatically chosen if equal to 0.0)\n"
        "          sY      = kernel bandwidth for Y (automatically chosen if equal to 0.0)\n"
        "          nrperm  = |nrperm| is the number of permutations used for estimating the p-value\n"
        "                    (if > 0,  use original biased HSIC estimator,\n"
        "                     if == 0, use gamma approximation,\n"
        "                     if < 0,  use unbiased HSIC estimator)\n"
        "\n"
        "OUTPUT:   p       = p-value of the HSIC\n"
        "                    (large p means independence, small p means dependence)\n"
        "          HSIC    = Hilbert Schmidt Independence Criterion estimator for X and Y\n"
        "          prob    = probability density of the HSIC\n"
        "          logp    = log p-value of the HSIC\n"
        "          dHSICdY = gradient of HSIC with respect to Y\n"
        "\n"
        "For more information, see\n"
        "\n"
        "  [1] Gretton, A., K. Fukumizu, C. H. Teo, L. Song, B. Schölkopf and A. J. Smola:\n"
        "  A Kernel Statistical Test of Independence. Advances in Neural Information\n"
        "  Processing Systems 20: Proceedings of the 2007 Conference, 585-592. (Eds.)\n"
        "  Platt, J. C., D. Koller, Y. Singer, S. Roweis, Curran, Red Hook, NY, USA (09 2008)\n"
        "\n"
        "  [2] Song, L., A. J. Smola, A. Gretton, K. M. Borgwardt and J. Bedo:\n"
        "  Supervised Feature Selection via Dependence Estimation. Proceedings of the\n"
        "  24th Annual International Conference on Machine Learning (ICML 2007),\n"
        "  823-830. (Eds.) Ghahramani, Z. ACM Press, New York, NY, USA (06 2007)\n"
        "\n"
        "Copyright (c) 2008-2012  Joris Mooij  <j.mooij@cs.ru.nl>\n"
        "All rights reserved.  See the file LICENSE for the license terms\n"
        );
    } 

    if( !mxIsDouble(X_IN) )
        mexErrMsgTxt("x should be a dense matrix, with entries of type double\n");
    size_t N = mxGetM(X_IN);
    if( N <= 2 )
        mexErrMsgTxt("N (sample size) should be at least 2\n");
    size_t d1 = mxGetN(X_IN);
    if( !mxIsDouble(Y_IN) || (mxGetM(Y_IN) != N) )
        mexErrMsgTxt("y should have the same number of rows as x, and entries of type double\n");
    size_t d2 = mxGetN(Y_IN);
    double *x = (double *)mxGetPr(X_IN);
    double *y = (double *)mxGetPr(Y_IN);

    double sx = 0.0;
    double sy = 0.0;
    int nrperm = 0;
    if( nrhs > NR_IN ) {
        if( !mxIsDouble(SX_IN) || (mxGetM(SX_IN) != 1) || (mxGetN(SX_IN) != 1) )
            mexErrMsgTxt("sx should be a double scalar\n");
        sx = *((double *)mxGetPr(SX_IN));

        if( nrhs > NR_IN+1 ) {
            if( !mxIsDouble(SY_IN) || (mxGetM(SY_IN) != 1) || (mxGetN(SY_IN) != 1) )
                mexErrMsgTxt("sy should be a double scalar\n");
            sy = *((double *)mxGetPr(SY_IN));
        }

        if( nrhs > NR_IN+2 ) {
            if( (mxGetM(NRPERM_IN) != 1) || (mxGetN(NRPERM_IN) != 1) )
                mexErrMsgTxt("nrperm should be a scalar");
            nrperm = (int)(*((double *)mxGetPr(NRPERM_IN)));
        }
    }

    HSICresult result = calcHSIC( N, d1, d2, x, y, sx, sy, nrperm, nlhs >= 5 );

    // Hand over results to MATLAB
    if( nlhs >= 1 ) {
        P_OUT = mxCreateDoubleMatrix(1,1,mxREAL);
        *(mxGetPr(P_OUT)) = result.p_value;
    }

    if( nlhs >= 2 ) {
        HSIC_OUT = mxCreateDoubleMatrix(1,1,mxREAL);
        *(mxGetPr(HSIC_OUT)) = result.HSIC;
    }

    if( nlhs >= 3 ) {
        PROB_OUT = mxCreateDoubleMatrix(1,1,mxREAL);
        *(mxGetPr(PROB_OUT)) = result.prob0;
    }

    if( nlhs >= 4 ) {
        LOGP_OUT = mxCreateDoubleMatrix(1,1,mxREAL);
        *(mxGetPr(LOGP_OUT)) = result.log_p_value;
    }

    if( nlhs >= 5 ) {
        DHSICDY_OUT = mxCreateDoubleMatrix(N,1,mxREAL);
        double *dHSICdY = mxGetPr(DHSICDY_OUT);
        for( size_t i = 0; i < N; i++ )
            dHSICdY[i] = result.dHSICdY[i];
    }
    
    return;
}
