function result = cep_anm (X,Y,methodpars)
% function result = cep_anm (X,Y,methodpars)
%
% Wrapper that runs Additive Noise Model (ANM)-based causal discovery on a pair of data vectors (X,Y).
%
% It fits a GP to the data and evaluates the model fit using a specificed evaluation method,
% either using the HSIC independence test, or using a entropy estimators in order to bound 
% the mutual information between input and residuals, or by using a Bayesian or likelihood score
% that assumes Gaussianity.
%
% If methodpars.evaluation == 'pHSIC' (default):
%   It decides on the direction having the higher (logarithm of the) HSIC p-value.
%   As confidence value, it returns the maximum of the two (logarithms of the) HSIC p-values.
%   As score, it returns (-1 / confidence) * decision.
%
% If methodpars.evaluation == 'HSIC':
%   It decides on the direction having the lower HSIC value.
%   As confidence value, it returns the minimum of the two HSIC values.
%   As score, it returns 1 / (1 + conf) * decision.
%
% If methodpars.evaluation == 'entropy':
%   It uses entropy.m with the specified entropy estimator methodpars.entest to calculate as score
%     -entropy(inputs) - entropy(residuals)
%   The direction having the higher score is chosen, and the confidence value is the absolute difference between the scores.
%   As score, it returns S = -entropy(X) - entropy(residuals X->Y) + entropy(Y) + entropy(residuals Y->X)
%
% If methodpars.evaluation == 'FN': 
%   Bayesian model comparison is used, using an additive-noise GP model for the conditional 
%   and a Gaussian model for the marginal distributions, for both causal directions. 
%   The direction having the higher evidence is chosen, and the logarithm of the Bayes factor
%   is used as confidence value.
%   As score, it returns the difference in the log marginal likelihoods S = ln(p(X,Y|X->Y)) - ln(p(X,Y|Y->X))
%
% If methodpars.evaluation == 'Gauss':
%    Gaussian likelihood score is used for doing model comparison.
%   As score, it returns the difference in the log likelihoods S = -ln(var(X)) - ln(var(residuals X->Y)) + ln(var(Y)) + ln(var(residuals Y->X))
%
% If methodpars.evaluation == 'MML':
%   The GPI-AN method in the GPI paper is used.
%   As score, it returns the difference in the negative minimum message lengths S = -MML(X,Y|X->Y)) + MML(X,Y|Y->X))
%
% For more details, see:
%   J. M. Mooij, J. Peters, D. Janzing, J. Zscheischler, B. Schoelkopf
%   Distinguishing cause from effect using observational data: methods and benchmarks
%   arXiv:1412.3773v2, submitted to Journal of Machine Learning Research
%   http://arxiv.org/abs/1412.3773v2
%
% INPUT:   X                Sample from variable X (Nx1 matrix)
%          Y                Sample from variable Y (Nx1 matrix)
%          methodpars       Struct containing the parameters for the method
%            .nrperm          number of permutations to use with HSIC test,
%                             0 means gamma approximation (default: 0)
%            .FITC            if nonzero, uses this amount of points for FITC approximation (using linear grid)
%            .splitdata       whether to split the data into a training set (for regression) and a test set (for independence testing)
%            .evaluation      model selection criterion: 'pHSIC' (default), 'HSIC', 'entropy', 'FN', 'Gauss', or 'MML'
%            .bandwidths      bandwidths for HSIC kernel (default: [0,0])
%            .meanf           GP pars mean function (default: 'meanAffine')
%            .minimize        GP pars minimization function: either 'minimize' (default) or 'minimize_lbfgsb'
%            .entest          Entropy estimation method to use (see entropy.m for options)
%
% OUTPUT:  result           Struct that should contain the following fields:
%            X                copy of X
%            Y                copy of Y
%            methodpars       copy of methodpars
%            decision         decision about causal direction
%                               1    means X->Y
%                               -1   means Y->X
%                               0    means no preference for X->Y or Y->X
%                               NAN  means error
%            conf             Confidence that the decision is correct.
%                               (The precise meaning of this value depends on the method. 
%                               In general, the higher the confidence, the more certain 
%                               a method is. It should be -Inf in case decision == NAN.)
%            score            Score for the decision. The higher the score, the more likely
%                               that X->Y, the lower the score, the more likely that Y->X
%            time             Computation time used.
%            opt              Struct containing any other (partial) results.
%              forward          Results for hypothetical ANM model X->Y
%              backward         Results for hypothetical ANM model Y->X
%                             (See source code for details)
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  % prepare result struct
  result = struct;
  %result.X = X; (DISABLED TO SAVE DISK SPACE)
  %result.Y = Y; (DISABLED TO SAVE DISK SPACE)
  result.methodpars = methodpars;
  result.opt = struct;

  if size(X,2) ~= 1 || size(Y,2) ~= 1 % for now, all methods only support univariate X and Y
    result.decision = nan;
    result.conf = -Inf;
  else
    % start measuring computation time
    tic;

    % init (set paths)
    curpath = path;
    global CEP_PATH_UTIL;
    addpath(CEP_PATH_UTIL);
    global CEP_PATH_FIT;
    addpath(CEP_PATH_FIT);
    global CEP_PATH_FASTHSIC;
    addpath(CEP_PATH_FASTHSIC);
    global CEP_PATH_GPML;
    run([CEP_PATH_GPML '/startup']); 
    global CEP_PATH_ITE;
    run([CEP_PATH_ITE '/ITE_add_to_path.m']);

    % set default parameters
    if ~isfield(methodpars,'nrperm')
      methodpars.nrperm = 0;
    end
    if ~isfield(methodpars,'FITC')
      methodpars.FITC = 0;
    end
    if ~isfield(methodpars,'splitdata')
      methodpars.splitdata = 0;
    end
    if ~isfield(methodpars,'evaluation')
      methodpars.evaluation = 'pHSIC';
    end
    if ~isfield(methodpars,'meanf')
      methodpars.meanf = 'meanAffine';
    end
    if ~isfield(methodpars,'minimize')
      methodpars.minimize = 'minimize';
    end
    if ~isfield(methodpars,'bandwidths')
      methodpars.bandwidths = [0,0];
    end
    if ~isfield(methodpars,'entest')
      methodpars.entest = 'KSG';
    end

    % perform causal discovery
    % set GP pars
    gppars.meanf = methodpars.meanf;
    gppars.minimize = methodpars.minimize;
    % fit both forward and backward models
    forward = GP_plus_eval(X,Y,methodpars,gppars);
    backward = GP_plus_eval(Y,X,methodpars,gppars);

    % take decision
    decision = 0;
    if forward.conf > backward.conf
      decision = 1;
    elseif forward.conf < backward.conf
      decision = -1;
    end

    % calculate confidence (the higher, the more confidence)
    if strcmp(methodpars.evaluation,'pHSIC')
      conf = max([forward.conf, backward.conf]);
      %score = conf * decision;
      score = (-1.0 / conf) * decision;
    elseif strcmp(methodpars.evaluation,'HSIC')
      conf = max([forward.conf, backward.conf]);
      score = (1.0 / (1.0 - conf)) * decision;  % == 1 / (1 + minHSIC) * decision
    else
      conf = abs(forward.conf - backward.conf);
      score = forward.conf - backward.conf;
    end

    % save results
    result.time = toc;
    result.decision = decision;
    result.conf = conf;
    result.score = score;
    % save method-specific results in result.opt (DISABLED TO SAVE DISK SPACE)
    % result.opt.forward = forward;
    % result.opt.backward = backward;

    % restore path
    path(curpath);
  end

return


function results = GP_plus_eval(input,output,methodpars,gppars)

    % split data into train and test set, if requested
    if methodpars.splitdata
      [train,test] = cvfolds(length(input),2);
      train = train{1}; test = test{1};
    else
      train = [1:length(input)];
      test  = [1:length(input)];
    end

    % fit GP
    if methodpars.FITC ~= 0
      gppars.FITC = linspace(min(input),max(input),methodpars.FITC)';
    end
    results.train = train;
    results.test = test;
    results.result = train_gp(input(train),output(train),gppars);
    results.fit = results.result.Yfit;
    results.stddev_noise = sqrt(results.result.Yvar);

    % evaluate independence of input versus residuals
    results.predict = test_gp(input(test),results.result.model,gppars);

    if strcmp(methodpars.evaluation,'pHSIC') || strcmp(methodpars.evaluation,'HSIC')
      if methodpars.nrperm ~= 0
        [results.pHSIC,results.HSIC,~,results.logpHSIC] = fasthsic(input(test),output(test) - results.predict.Ytest,methodpars.bandwidths(1),methodpars.bandwidths(2),methodpars.nrperm);
      else
        [results.pHSIC,results.HSIC,~,results.logpHSIC] = fasthsic(input(test),output(test) - results.predict.Ytest,methodpars.bandwidths(1),methodpars.bandwidths(2));
      end
      if strcmp(methodpars.evaluation,'pHSIC')
        results.conf = results.logpHSIC;
      else
        results.conf = -results.HSIC;
      end
    elseif strcmp(methodpars.evaluation,'entropy')
      results.conf = -entropy(input(test),methodpars.entest) - entropy(output(test) - results.predict.Ytest, methodpars.entest);
    elseif strcmp(methodpars.evaluation,'FN') || strcmp(methodpars.evaluation,'Gauss') || strcmp(methodpars.evaluation,'MML')
      Ntest = length(test);

      % calculate description length of marginal model
      if strcmp(methodpars.evaluation,'FN') || strcmp(methodpars.evaluation,'Gauss')
        results.nlevid_marg = Ntest * (0.5 * log(2*pi) + 0.5 + log(std(input(test),1)));
      elseif strcmp(methodpars.evaluation,'MML')
        % init (set paths)
        curpath = path;
        global CEP_PATH_GPI;
        addpath(CEP_PATH_GPI);
        global CEP_PATH_MMLMIXTURES;
        addpath(CEP_PATH_MMLMIXTURES);

        % add field MML if it is missing
        if ~isfield(methodpars,'MML')
          methodpars.MML = struct;
        end
        % set default MML values
        if ~isfield(methodpars.MML,'reg')
          methodpars.MML.reg = 1e-4;
        end

        % run MML on input distribution
        [results.mml,results.info_X] = mmlgmm(input(test),methodpars.MML);

        results.nlevid_marg = results.mml;

        % restore path
        path(curpath);
      end

      % calculate description length of conditional model
      if strcmp(methodpars.evaluation,'Gauss')
        results.nlevid_cond = Ntest * (0.5 * log(2*pi) + 0.5 + log(std(output(test) - results.predict.Ytest,1)));
      else
        results.nlevid_cond = results.result.loss;
      end

      % calculate total description length
      results.conf = -(results.nlevid_marg + results.nlevid_cond);
    else
      error('Unknown evaluation method');
    end

return
