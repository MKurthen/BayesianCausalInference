function result = cep_count (X,Y,methodpars)
% function result = cep_count (X,Y,methodpars)
%
% Simply counts the number of unique (x,y) values and decides the causal direction based on that:
% if X has less unique values than Y, than infer that X causes Y. 
%
% As confidence value, it returns the absolute difference in the number of unique values, 
% normalized by the maximum number of unique values.
%
% This is mainly useful for checking whether there is some asymmetric bias in the data that
% happens to correlate with the causal direction.
%
% INPUT:   X                Sample from variable X (Nx1 matrix)
%          Y                Sample from variable Y (Nx1 matrix)
%          methodpars       Struct containing the parameters for the method
%                             (This method takes no parameters)
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
%                               (See source code for details)
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

    % perform causal discovery based on counting unique values
    f1 = length(unique(X));
    f2 = length(unique(Y));
    f = f1 - f2;

    % take decision
    if f < 0
      decision = 1;
    elseif f > 0
      decision = -1;
    else
      decision = 0;
    end

    % calculate confidence (the higher, the more confidence)
    conf = abs(f) / max([f1,f2]);

    % save results
    result.time = toc;
    result.decision = decision;
    result.conf = conf;
    result.score = -f / max([f1,f2]);
    % save method-specific results in result.opt (DISABLED TO SAVE DISK SPACE)
    result.opt.f = f;
    result.opt.f1 = f1;
    result.opt.f2 = f2;

    % restore path
    path(curpath);
  end

return
