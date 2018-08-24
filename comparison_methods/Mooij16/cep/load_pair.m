function [cause,effect,weight] = load_pair (pair,preprocessing,datadir)
% function [cause,effect,weight] = load_pair (pair,preprocessing,datadir)
%
% Loads a cause-effect pair and applies preprocessing to the data
%
% INPUT:   pair           number of the cause effect pair (e.g., 1)
%          preprocessing  the preprocessing to apply to the data
%            .normalize     false:  do not normalize data
%                           true:   make mean == 0 and std == 1 (default: true)
%            .maxN          maximum number of data points for subsampling,
%                           0 for all available data points (default: inf)
%            .randseed      random number seed for permuting data points (default: 42)
%            .undisc        whether the data should be 'undiscretized'
%                             (which adds small amounts of noise to the
%                              components of data points whose values
%                              coincide with those of other data points) (default: false)
%            .disc          whether the data should be 'discretized'
%                             (which uses kmeans clustering on one of the two
%                              variables to ensure that both have the same
%                              number of unique values) (default: false)
%            .disturbance   standard deviation of Gaussian noise added to
%                           the normalized, undiscretized variables (default: 0)
%          datadir        directory path to cause-effect-pairs
%
% OUTPUT:  cause          preprocessed cause data (N*dX vector)
%          effect         preprocessed effect data (N*dY vector)
%          weight         weight of the pair in overall accuracy score
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  curpath = path;
  global CEP_PATH_UTIL;
  addpath(CEP_PATH_UTIL);

  metadata = load(sprintf('%s/pairmeta.txt',datadir));
  if metadata(pair,1) ~= pair 
    error('Assumption on metadata not satisfied');
  else
    % set default parameters
    if nargin < 2
      preprocessing = struct;
    end
    if ~isfield(preprocessing,'normalize')
      preprocessing.normalize = true;
    end
    if ~isfield(preprocessing,'maxN')
      preprocessing.maxN = inf;
    end
    if ~isfield(preprocessing,'randseed')
      preprocessing.randseed = 42;
    end
    if ~isfield(preprocessing,'undisc')
      preprocessing.undisc = false;
    end
    if ~isfield(preprocessing,'disc')
      preprocessing.disc = false;
    end
    if ~isfield(preprocessing,'disturbance')
      preprocessing.disturbance = false;
    end

    % load data
    filename = sprintf('%s/pair%04d.txt',datadir,pair);
    data = load('-ascii',filename);

    % assign cause and effect
    cause = data(:,metadata(pair,2):metadata(pair,3));
    effect = data(:,metadata(pair,4):metadata(pair,5));

    % normalize cause and effect
    if preprocessing.normalize
      cause = normalize(cause);
      effect = normalize(effect);
    end
    
    % undiscretize
    if preprocessing.undisc
      cause = undiscretize(cause);
      effect = undiscretize(effect);
    end

    % apply deterministic random permutation
    N = size(data,1);
    index = myrandperm(N,preprocessing.randseed);
    % take only first N data points, if necessary
    if N > preprocessing.maxN
      index = index(1:preprocessing.maxN);
    end
    cause = cause(index,:);
    effect = effect(index,:);
    weight = metadata(pair,6);

    % discretize the variable with more unique values
    if preprocessing.disc && size(cause,2) == 1 && size(effect,2) == 1
      Ncause = length(unique(cause,'rows'));
      Neffect = length(unique(effect,'rows'));
      Nvals = min(Ncause,Neffect);
      if Ncause > Nvals
        cause = discretize(cause,Nvals);
      elseif Neffect > Nvals
        effect = discretize(effect,Nvals);
      end
      assert(length(unique(cause,'rows')) == length(unique(effect,'rows')));
    end

    % add disturbance
    cause = cause + preprocessing.disturbance * randn(size(cause));
    effect = effect + preprocessing.disturbance * randn(size(effect));
  end

  % restore path
  path(curpath);
return
