function result = train_gp(X, Y, pars)
% function result = train_gp(X, Y, pars)
%
% Trains a Gaussian Process with RBF kernel regression model.
% See also test_gp, train_template.
%
% INPUT:
%   X         Nxd matrix of training inputs (N data points, d dimensions)
%   Y         Nx1 matrix of training outputs (N data points)
%   pars      structure containing parameters of the regression method
%     .meanf    mean function: one of {'meanZero','meanConst','meanAffine'}
%     .covf     covariance function: one of {'covSEiso','covSEard'}
%     .FITC     if nonempty, inducing inputs for FITC approximation;
%               should be K*d matrix (default: [])
%     .lik      one of {'likGauss', 'likLaplace', 'likLogistic', 'likT'}
%               default: likGauss
%     .inf      one of {'infExact','infLaplace','infEP','infVB'}
%               default: infExact
%     .maxiter  maximum number of conjugate gradient steps (default: 1000)
%     .hyp      initial hyperparameters (optional)
%     .minimize one of {'minimize','minimize_lbfgsb'}
%               default: minimize_lbfgsb
%
% OUTPUT:
%   result    structure with the result of the regression
%               .model        learned model:
%                 .hyp          optimized hyperparameters
%                 .X            train inputs
%                 .Y            train outputs
%               .Yfit         expected outputs for training inputs according to the learned model
%               .eps          noise values (residuals)
%               .loss         loss function of trained model
%               .Yvar         variance of training outputs according to the learned model
%
% NOTE
%   Uses GPML 3.4 code (which should be in the matlab path)
%
% Copyright (c) 2008-2011  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  % if X is empty, return Y as residuals
  if size(X,2)==0
    result.Yfit = zeros(size(Y,1),1);
    result.eps = Y;
    return
  end

  % check input arguments
  if size(Y,2)~=1 | size(X,1)~=size(Y,1)
    error('X should be Nxd and Y should be Nx1');
  end

  % set default parameters
  if ~isfield(pars,'covf')
    pars.covf = 'covSEiso';
  end;
  if ~isfield(pars,'meanf')
    pars.meanf = 'meanZero';
  end
  if ~isfield(pars,'FITC')
    pars.FITC = [];
  end
  if ~isfield(pars,'lik')
    pars.lik = 'likGauss';
  end
  if ~isfield(pars,'inf')
    if pars.FITC ~= 0
      pars.inf = 'infFITC';
    else
      pars.inf = 'infExact';
    end
  end
  if ~isfield(pars,'maxiter')
    pars.maxiter = 1000;
  end
  if ~isfield(pars,'minimize')
    pars.minimize = 'minimize_lbfgsb';
  end

  % initialize hyperparameters, if none are provided
  if isfield(pars,'hyp')
    hyp = pars.hyp;
  else
    sf  = std(Y);  % sigma_function
    ell = std(X)';  % length scale
    sn  = 0.1 * sf;  % sigma_noise
    if strcmp(pars.covf, 'covSEiso')
      hyp.cov = log([sqrt(sum(ell.^2));sf]);
    elseif strcmp(pars.covf, 'covSEard')
      hyp.cov = log([ell;sf]);
    else
      error('Unknown covariance function');
    end

    if strcmp(pars.meanf,'meanZero')
      % m(x) = 0
    elseif strcmp(pars.meanf,'meanConst')
      b = 0.0;
      hyp.mean = b; % m(x) = b
    elseif strcmp(pars.meanf,'meanAffine')
      a = zeros(size(X,2),1);
      b = 0.0;
      hyp.mean = [a;b]; % m(x) = a*x+b
    else
      error('Unknown mean function');
    end

    if strcmp(pars.lik,'likT')
      nu = 4;
      hyp.lik = log([nu-1;sqrt((nu-2)/nu)*sn]);
    else
      hyp.lik = log(sn);
    end
  end

  % set mean function
  if strcmp(pars.meanf,'meanZero')
    meanf = {@meanZero};
  elseif strcmp(pars.meanf,'meanConst')
    meanf = {@meanConst};
  elseif strcmp(pars.meanf,'meanAffine')
    meanf = {@meanSum,{@meanLinear,@meanConst}};
  else
    error('Unknown mean function');
  end

  % set covariance function
  if strcmp(pars.covf, 'covSEiso')
    covf = {@covSEiso};
  elseif strcmp(pars.covf, 'covSEard')
    covf = {@covSEard};
  else
    error('Unknown covariance function');
  end

  if ~isempty(pars.FITC)
    warning('Using FITC approximation');
    covf = {@covFITC, covf, pars.FITC};
  end

  % learn hyperparameters
  if strcmp(pars.minimize,'minimize')
    hyp = minimize(hyp,'gp',-pars.maxiter,pars.inf,meanf,covf,pars.lik,X,Y);
  elseif strcmp(pars.minimize,'minimize_lbfgsb')
    try
      hyp = minimize_lbfgsb(hyp,'gp',-pars.maxiter,pars.inf,meanf,covf,pars.lik,X,Y);
    catch exception % fall back on minimize
      warning('Caught exception from minimize_lbfgsb. Falling back on minimize!');
      hyp = minimize(hyp,'gp',-pars.maxiter,pars.inf,meanf,covf,pars.lik,X,Y);
    end
    % may throw MatlabException("Solver unable to satisfy convergence \ criteria due to abnormal termination")
    % should add try...catch exception...end and fallback to minimize
  else
    error('Unknown minimization function');
  end

  % calculate evidence (log marginal likelihood)
  lml = -gp(hyp,pars.inf,meanf,covf,pars.lik,X,Y);

  % calculate fit on training set and assign result fields
  result = struct;
  [result.Yfit,result.Yvar] = gp(hyp,pars.inf,meanf,covf,pars.lik,X,Y,X);  % predict on X
  result.eps = Y - result.Yfit;
  result.loss = -lml;
  result.model = struct;
  result.model.hyp = hyp;
  result.model.X = X;
  result.model.Y = Y;
return
