function result = cep_igci (X,Y,methodpars)
% function result = cep_igci (X,Y,methodpars)
%
% Wrapper that runs Information Geometric Causal Inference (IGCI) on a pair of data vectors (X,Y).
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
%            .refMeasure      1 = uniform, 2 = Gaussian
%            .estimator       {'org_entropy','slope','slope++','asymmetric','entropy'}
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
%                               (See source code for details).
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
    global CEP_PATH_ITE;
    run([CEP_PATH_ITE '/ITE_add_to_path.m']);
    global CEP_PATH_UTIL;
    addpath(CEP_PATH_UTIL); 
    global CEP_PATH_IGCI;
    addpath(CEP_PATH_IGCI); 

    % set default parameters
    if ~isfield(methodpars,'estimator')
      methodpars.estimator = 'org_entropy';
    end
    if ~isfield(methodpars,'refMeasure')
      methodpars.refMeasure = 1;
    end

    % perform IGCI
    if strcmp(methodpars.estimator,'org_entropy')
      f = igci(X,Y,methodpars.refMeasure,1);
      f1 = nan;
      f2 = nan;
      conf = abs(f);
    elseif strcmp(methodpars.estimator,'slope')
      f = igci(X,Y,methodpars.refMeasure,2);
      f1 = nan;
      f2 = nan;
      conf = abs(f);
    elseif strcmp(methodpars.estimator,'slope++')
      f = igci(X,Y,methodpars.refMeasure,3);
      f1 = nan;
      f2 = nan;
      conf = abs(f);
    elseif strcmp(methodpars.estimator,'asymmetric')
      [f1, f2] = igci2(X,Y,methodpars.refMeasure);
      forward.conf = -abs(f1);
      backward.conf = -abs(f2);
      f = backward.conf - forward.conf;
%      conf = max([forward.conf, backward.conf]);
      conf = abs(forward.conf - backward.conf);
    elseif strcmp(methodpars.estimator,'entropy')
      f1 = nan;
      f2 = nan;
      f = -entropy(normalize(X,methodpars.refMeasure),methodpars.entest) + entropy(normalize(Y,methodpars.refMeasure),methodpars.entest); % -S(X) + S(Y)
      conf = abs(f);
    end

    % take decision
    if f < 0
      decision = 1;
    elseif f > 0
      decision = -1;
    else
      decision = 0;
    end

    % save results
    result.time = toc;
    result.decision = decision;
    result.conf = conf;
    result.score = -f;
    % save method-specific results in result.opt (DISABLED TO SAVE DISK SPACE)
    % result.opt.f = f;
    % result.opt.f1 = f1;
    % result.opt.f2 = f2;

    % restore path
    path(curpath);
  end

return
