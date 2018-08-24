function f = randGP(X,sigma_in,sigma_out,sigma_noise)
% function f = randGP(X,sigma_in,sigma_out,sigma_noise)
%
% Draws from a random Gaussian Process with covSEard covariance function
%
% INPUT:
%   X            data points (Nxd)
%   sigma_in     bandwidth parameter for GP input
%   sigma_out    standard-deviation of GP output
%   sigma_noise  standard-deviation of additive white noise
%
% OUTPUT:
%   f            random function+noise values of the random GP (Nx1) 
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  % init (set paths)
  curpath = path;
  global CEP_PATH_GPML;
  run([CEP_PATH_GPML '/startup']); 

  % number of data points
  N=size(X,1);
  % dimensionality
  d=size(X,2);

  % calculate covariance matrix
  K = covSEard(log([sigma_in';sigma_out]), X);

  % calculate its Cholesky decomposition (C'*C = S)
  C = chol(K + sigma_noise^2 * eye(N));

  % sample from GP prior
  f = C' * randn(N,1);

  % restore path
  path(curpath);

return
