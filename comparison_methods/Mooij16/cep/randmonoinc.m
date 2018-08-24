function monoincf = randmonoinc(X,sigma_in,sigma_out,sigma_noise)
% function monoincf = randmonoinc(X,sigma_in,sigma_out,sigma_noise)
%
% Draws a sample from a random density by sampling from a GP with
% covSEard covariance function, and then taking the cumulative 
% integral of its exponential. 
%
% INPUT:
%   X            data points (Nxd)
%   sigma_in     bandwidth parameter for GP input
%   sigma_out    standard-deviation of GP output
%   sigma_noise  standard-deviation of additive white noise
%
% OUTPUT:
%   monoincf     sample from random density
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

% draw random gp
f = randgp(X,sigma_in,sigma_out,sigma_noise);

% take exp
ef = exp(f);

% sort X
[dum,idx] = sort(X);

% calculate cumulative integral
monoincf = zeros(size(X,1),1);
monoincf(idx) = cumtrapz(X(idx),ef(idx))';

return
