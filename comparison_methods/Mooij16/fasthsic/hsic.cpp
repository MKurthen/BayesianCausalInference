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


#define USE_BOOST 1


#include "hsic.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <limits>
#ifdef USE_BOOST
    #include <boost/math/distributions/gamma.hpp>
    #include <boost/math/special_functions/gamma.hpp>
#else
    #include <gsl/gsl_sf_gamma.h>
    #include <gsl/gsl_errno.h>
    #include <gsl/gsl_randist.h>
#endif


using namespace std;


#ifndef NAN
const double NAN = std::numeric_limits<double>::quiet_NaN();
#endif


void get_norm( double *data, size_t N, size_t dims, std::vector<double> &result ) {
    result.resize( N * N );
    if( dims == 1 ) {  // optimized version of dims > 1 case
        for( size_t i = 0; i < N; i++ ) {
           for( size_t j = 0; j < i; j++ ) {
               double x = (data[i] - data[j]) * (data[i] - data[j]);
               result[i*N+j] = x;
               result[j*N+i] = x;
           }
           result[i*N+i] = 0.0;
        }
    } else {
        for( size_t i = 0; i < N; i++ ) {
           for( size_t j = 0; j < i; j++ ) {
               double x = 0.0;
               for( size_t k = 0; k < N*dims; k += N ) {
                   double x_k = data[i+k] - data[j+k];
                   x += x_k * x_k;
               }
               result[i*N+j] = x;
               result[j*N+i] = x;
           }
           result[i*N+i] = 0.0;
        }
    }
}


double get_sigma( const std::vector<double> &norm, size_t N ) {
    double med = 0.0;
    std::vector<double> x;
    x.reserve( N * (N-1) / 2 );
    for( size_t i = 0; i < N; i++ )
        for( size_t j = i+1; j < N; j++ )
            if( norm[i*N+j] != 0.0 )
                x.push_back( norm[i*N+j] );
    nth_element( x.begin(), x.begin() + x.size() / 2, x.end() );
    double x1 = *(x.begin() + x.size() / 2);
    if( x.size() % 2 ) {
        med = x1;
    } else {
        nth_element( x.begin(), x.begin() + x.size() / 2 - 1, x.end() );
        double x2 = *(x.begin() + x.size() / 2 - 1);
        med = (x1 + x2) / 2.0;
    }

    return sqrt( 0.5 * med );
}


HSICresult calcHSIC( size_t N, size_t dims_x, size_t dims_y, double *x, double *y, double sigma_x, double sigma_y, int nrperm, bool calc_dHSICdY ) {
    // build matrices x_norm, y_norm (containing the squared l2-distances between data points)
    std::vector<double> x_norm;
    get_norm( &(x[0]), N, dims_x, x_norm );
    std::vector<double> y_norm;
    get_norm( &(y[0]), N, dims_y, y_norm );

    // if necessary, choose kernel bandwidth using heuristic
    if( sigma_x == 0.0 )
        sigma_x = get_sigma( x_norm, N );
    if( sigma_y == 0.0 )
        sigma_y = get_sigma( y_norm, N );

    // build RBF kernel matrix Kx
    double c = -0.5 / (sigma_x * sigma_x);
    std::vector<double> Kx;
    Kx.reserve( N*N );
    double Kx_sum = 0.0;
    std::vector<double> Kx_sums(N,0.0);
    for( size_t i = 0; i < N; i++ ) {
        for( size_t j = 0; j < N; j++ ) {
            double x;
            if( c != 0.0 ) // use RBF kernel
                x = exp( x_norm[i*N+j] * c );
            else // use distance matrix
                x = sqrt(x_norm[i*N+j]);
            Kx[i*N+j] = x;
            Kx_sums[i] += x;
        }
        Kx_sum += Kx_sums[i];
    }

    // build RBF kernel matrix Ky
    double eps = 1e-6;
    c = -0.5 / (sigma_y * sigma_y);
    std::vector<double> Ky;
    Ky.reserve( N*N );
    double Ky_sum = 0.0;
    std::vector<double> Ky_sums(N,0.0);
    for( size_t i = 0; i < N; i++ ) {
        for( size_t j = 0; j < N; j++ ) {
            double y;
            if( c != 0.0 ) // use RBF kernel
                y = exp( y_norm[i*N+j] * c );
            else // use distance matrix
                y = sqrt(y_norm[i*N+j] + eps);
            Ky[i*N+j] = y;
            Ky_sums[i] += y;
        }
        Ky_sum += Ky_sums[i];
    }

    HSICresult result;
    if( nrperm < 0 ) { // unbiased HSIC estimate with permutation test (see [2])
        double KxKysum = 0.0;
        double tr_KxKy = 0.0;
        for( size_t i = 0; i < N; i++ ) {
            KxKysum += Kx_sums[i] * Ky_sums[i];
            for( size_t j = 0; j < N; j++ )
                tr_KxKy += Kx[i*N+j] * Ky[i*N+j];
        }
        result.HSIC = (tr_KxKy + (Kx_sum * Ky_sum) / ((N-1) * (N-2)) - 2.0 / (N-2) * KxKysum) / (N * (N-3));

        // construct random permutation
        vector<size_t> permutation(N, 0);
        for( size_t i = 0; i < N; i++ )
            permutation[i] = i;

        size_t count = 0;  // counts how often permuted HSIC is larger than HSIC
        for( size_t perm = 0; perm < abs(nrperm); perm++ ) {
            random_shuffle( permutation.begin(), permutation.end() );

            double ptr_KxKy = 0.0;
            double pKxKysum = 0.0;
            for( size_t i = 0; i < N; i++ ) {
                size_t pi = permutation[i];

                pKxKysum += Kx_sums[i] * Ky_sums[pi];
                for( size_t j = 0; j < N; j++ )
                    ptr_KxKy += Kx[i*N+j] * Ky[pi*N+permutation[j]];
            }
            double pHSIC = (ptr_KxKy + (Kx_sum * Ky_sum) / ((N-1) * (N-2)) - 2.0 / (N-2) * pKxKysum) / (N * (N-3));

            if( pHSIC > result.HSIC )
                count++;
        }

        result.p_value = (count + 1.0) / (abs(nrperm) + 2.0);  // Incorporate a little prior to prevent p_values of 0 or 1
        result.log_p_value = std::log( result.p_value );
        assert( !calc_dHSICdY );
    } else if( nrperm == 0 ) { // use biased HSIC estimator with gamma approximation for p-value (see [1])
        double tr_KxHKyH = 0.0;
        double tr_KxHKxH = 0.0;
        double tr_KyHKyH = 0.0;
        for( size_t i = 0; i < N; i++ )
            for( size_t j = 0; j < N; j++ ) {
                double KxH_ij = Kx[i*N + j] - Kx_sums[i] / N;    // (KH)_{ij}
                double KxH_ji = Kx[i*N + j] - Kx_sums[j] / N;    // (KH)_{ji}
                double KyH_ij = Ky[i*N + j] - Ky_sums[i] / N;    // (LH)_{ij}
                double KyH_ji = Ky[i*N + j] - Ky_sums[j] / N;    // (LH)_{ji}
                tr_KxHKyH += KxH_ij * KyH_ji;
                tr_KxHKxH += KxH_ij * KxH_ji;
                tr_KyHKyH += KyH_ij * KyH_ji;
            }

        double HSIC = tr_KxHKyH / (N * N);

        double x_mu = 1.0 / (N*(N-1)) * (Kx_sum - N);
        double y_mu = 1.0 / (N*(N-1)) * (Ky_sum - N);
        double mean_H0 = (1.0 + x_mu * y_mu - x_mu - y_mu) / N;
        double var_H0 = (2.0 * (N-4) * (N-5)) / (N * (N-1.0) * (N-2) * (N-3) * pow(N-1,4.0)) * tr_KxHKxH * tr_KyHKyH;

        double a = mean_H0 * mean_H0 / var_H0;
        double b = N * var_H0 / mean_H0;

        result.HSIC = HSIC;
        result.HSIC0_mean = mean_H0;
        result.HSIC0_var = var_H0;

#ifdef USE_BOOST
        try {
            result.p_value = boost::math::gamma_q( a, N * HSIC / b );
            result.prob0 = boost::math::pdf( boost::math::gamma_distribution<double>(a, b), N * HSIC );
            if( result.p_value == 0.0 ) {
                // Underflow occurred. Deal with this using the solution proposed on http://math.stackexchange.com/questions/1041674/log-of-1-reguralized-incomplete-gamma-function-upper 
                double x = N * HSIC / b;
                double log_p_value = -x + a * std::log(x) - boost::math::lgamma(a) + std::log(1.0 / (x + (1.0 - a) / (1.0 + 1.0 / (x + (2.0 - a) / (1.0 + 2.0 / x)))));
                result.log_p_value = log_p_value;
            } else
                result.log_p_value = std::log( result.p_value );
        } catch( const exception& e ) {
            result.p_value = NAN;
            result.prob0 = NAN;
        }
#else
        gsl_sf_result gslresult;
        gsl_error_handler_t *old_handler = gsl_set_error_handler_off();     // disable standard GSL error handler
        int status = gsl_sf_gamma_inc_Q_e( a, N * HSIC / b, &gslresult );   // 'upper' incomplete gamma function; gsl_sf_gamma_inc_P_e is the 'lower' one
        if( status ) {
            result.p_value = NAN;
        } else
            result.p_value = gslresult.val;
        result.log_p_value = std::log( result.p_value );
        result.prob0 = gsl_ran_gamma_pdf (N * HSIC, a, b);
        gsl_set_error_handler (old_handler);                                // enable standard GSL error handler
#endif

        if( calc_dHSICdY ) {
            result.dHSICdY = vector<double>( N );
            if( 1.0 / sigma_y != 0.0 ) { // use RBF kernel
                for( size_t i = 0; i < N; i++ ) {
                    for( size_t j = 0; j < N; j++ ) {
                        double HKxH_ij = Kx[i*N + j] - Kx_sums[i] / N - Kx_sums[j] / N + Kx_sum / (N * N);  // Kbar[i*N + j]
                        result.dHSICdY[i] += HKxH_ij * Ky[i*N + j] * (y[j] - y[i]);
                    }
                    result.dHSICdY[i] *= 2.0 / (N * N * sigma_y * sigma_y);
                }
            } else { // use distance matrix
                for( size_t i = 0; i < N; i++ ) {
                    for( size_t j = 0; j < N; j++ ) {
                        double HKxH_ij = Kx[i*N + j] - Kx_sums[i] / N - Kx_sums[j] / N + Kx_sum / (N * N);  // Kbar[i*N + j]
                        result.dHSICdY[i] += HKxH_ij / Ky[i*N+j] * (y[j] - y[i]);
                    }
                    result.dHSICdY[i] *= -2.0 / (N * N);
                }
            }
        }
    } else { // biased HSIC estimate with permutation test (see [1])
        // calculate HSIC and precalculate KxH, HKy
        double tr_KxHKyH = 0.0;
        vector<double> KxH, HKy;
        KxH.reserve( N*N );
        HKy.reserve( N*N );
        for( size_t i = 0; i < N; i++ )
            for( size_t j = 0; j < N; j++ ) {
                double KxH_ij = Kx[i*N + j] - Kx_sums[i] / N;
                double KyH_ji = Ky[i*N + j] - Ky_sums[j] / N;
                KxH.push_back( KxH_ij );
                HKy.push_back( KyH_ji );
                tr_KxHKyH += KxH_ij * KyH_ji;
            }
        result.HSIC = tr_KxHKyH / (N * N);

        // construct random permutation
        vector<size_t> permutation(N, 0);
        for( size_t i = 0; i < N; i++ )
            permutation[i] = i;

        size_t count = 0;  // counts how often permuted HSIC is larger than HSIC
        for( size_t perm = 0; perm < nrperm; perm++ ) {
            random_shuffle( permutation.begin(), permutation.end() );

            double ptr_KxHKyH = 0.0;
            for( size_t i = 0; i < N; i++ ) {
                size_t pi = permutation[i];
                for( size_t j = 0; j < N; j++ )
                    ptr_KxHKyH += KxH[i*N + j] * HKy[pi*N + permutation[j]];
            }

            if( ptr_KxHKyH > tr_KxHKyH )
                count++;
        }

        result.p_value = (count + 1.0) / (nrperm + 2.0);  // Incorporate a little prior to prevent p_values of 0 or 1
        result.log_p_value = std::log( result.p_value );

        assert( !calc_dHSICdY );
    }
    return result;
}
