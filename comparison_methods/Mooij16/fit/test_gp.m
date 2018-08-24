function result = test_gp(Xtest, model, pars)
% function result = test_gp(Xtest, model, pars)
%
% Evaluate a (trained) Gaussian Process with RBF kernel on test data Xtest.
% See also: train_gp, test_template.
%
% INPUT:
%   Xtest     Nxd matrix of test inputs (N data points, d dimensions)
%   model     the model:
%               .hyp          optimized hyperparameters
%               .X            train inputs (Ntrain x d matrix)
%               .Y            train outputs (Ntrain x 1 matrix)
%   pars      structure containing parameters of the regression method
%     .meanf    mean function: one of {'meanZero','meanConst','meanAffine'}
%     .covf     covariance function: one of {'covSEiso','covSEard'}
%     .FITC     if nonempty, inducing inputs for FITC approximation;
%               should be K*d matrix (default: [])
%     .lik      one of {'likGauss', 'likLaplace', 'likLogistic', 'likT'}
%               default: likGauss
%     .inf      one of {'infExact','infLaplace','infEP','infVB'}
%               default: infExact
%
% OUTPUT:
%   result    structure with the result of the regression
%               .Ytest        Nx1 matrix: expected outputs of the model,
%                             evaluated on test inputs
%               .loss         loss function of model on test inputs
%               .Ytestvar     Nx1 matrix: variances of outputs of the model,
%                             evaluated on test inputs
%
% NOTE
%   Uses GPML 3.4 code (which should be in the matlab path)
%
% Copyright (c) 2008-2011  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  % check input arguments
  if size(Xtest,2) ~= size(model.X,2)
    error('Xtest and model.X should have same number of columns');
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
    covf = {@covFITC, covf, pars.FITC};
  end


  % calculate prediction of model on test set
  result = struct;
  [result.Ytest,result.Ytestvar] = gp(model.hyp,pars.inf,meanf,covf,pars.lik,model.X,model.Y,Xtest);

  % calculate negative evidence for test output
  result.loss = gp(model.hyp,pars.inf,meanf,covf,pars.lik,Xtest,result.Ytest);
return
