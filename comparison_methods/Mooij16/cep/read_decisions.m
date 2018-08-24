function [decisions,confidences,ranks,scores,times] = read_decisions(outdir,filteredpairs,strict,allownans)
% function [decisions,confidences,ranks,scores,times] = read_decisions(outdir,filteredpairs,strict,allownans)
%
% Reads the results of various methods from outdir and ranks them according to confidence
%
% INPUT:
%   outdir                directory from which to read the results
%   filteredpairs         vector of cause-effect pairs to consider
%   strict                if strict == 0, ignore any errors (non-existing files, etc.)
%   allownans             if allownans == 0, nans in decisions/confidences will be replaced by nondecisions
%
% OUTPUT:
%   decisions             vector of decisions (-1 or +1) for each pair
%   confidences           vector of confidence of the decisions for each pair
%   ranks                 vector of ranks of pairs according to confidence
%                           (ranks(1) being the most confident pair, ranks(end) the least one)
%   scores                vector of scores for each pair
%   times                 vector of computation times
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  fprintf('Reading method %s\n', outdir);

  % read decisions and confidences
  nrpairs = length(filteredpairs);
  alpha = zeros(nrpairs,1);
  decisions = zeros(nrpairs,1);
  confidences = zeros(nrpairs,1);
  scores = zeros(nrpairs,1);
  times = zeros(nrpairs,1);
  confs_forward = zeros(nrpairs,1);
  confs_backward = zeros(nrpairs,1);
  for pairid=1:nrpairs
    pair = filteredpairs(pairid);
    filename = sprintf('%s/pair%04d.hdf5',outdir,pair);
    if exist(filename,'file')
      fprintf('Reading %s\n',filename);
      result = load('-mat',filename);
%     result = load('-hdf5',filename);  % FOR OCTAVE
      if isfield(result,'result') 
        % NEW INTERFACE
        result = result.result;
      end
      if ~isfield(result,'decision') || ~isfield(result,'conf')
        error(sprintf('%s does not have the required fields',filename));
      end
%      if isfield(result,'forward') && isfield(result,'backward')
%        if isfield(result.forward,'conf') && isfield(result.backward,'conf')
%          confs_forward(pairid) = result.forward.conf;
%          confs_backward(pairid) = result.backward.conf;
%
%%TODO: this seems to work great for ANM-pHSIC!
%if 0
%  warning('Changing decisions!');
%  if max(result.forward.conf,result.backward.conf) > 0.01 && min(result.forward.conf,result.backward.conf) < 0.01
%    if result.forward.conf > result.backward.conf
%      result.decision = 1;
%    else
%      result.decision = -1;
%    end
%    result.confidence = 1 - min(result.forward.conf,result.backward.conf);
%  else
%    result.decision = 0;
%    result.confidence = -Inf;
%  end
%end
%
%        else
%          confs_forward(pairid) = nan;
%          confs_backward(pairid) = nan;
%        end
%      end
      decisions(pairid) = result.decision;
      confidences(pairid) = result.conf;
      if isfield(result,'score')
        scores(pairid) = result.score;
      else
        scores(pairid) = nan;
      end
      times(pairid) = result.time;

      if decisions(pairid) ~= 0 && decisions(pairid) ~= 1 && decisions(pairid) ~= -1 && ~isnan(decisions(pairid))
        error(sprintf('Invalid decision value in %s',filename));
      end
      if isnan(confidences(pairid)) || isnan(decisions(pairid))
        warning(sprintf('NAN in decision or confidence value in %s',filename));
      end
      if decisions(pairid) == 0 % default: lowest confidence
        confidences(pairid) = -Inf;
      end
    else
      if strict
        error(sprintf('Result file %s does not exist',filename));
      else
        warning(sprintf('Result file %s does not exist',filename));
        decisions(pairid) = nan;
        confidences(pairid) = nan;
      end
    end
    if ~allownans
      if isnan(decisions(pairid))
        decisions(pairid) = 0;
        confidences(pairid) = -Inf;
      end
      if isnan(confidences(pairid))
        confidences(pairid) = -Inf;
      end
      if isnan(scores(pairid))
        scores(pairid) = 0.0;
      end
    end
  end

  % calculate ranks
  [dummy,idx] = sort(confidences,1,'descend');
  ranks = zeros(nrpairs,1);
  for i = 1:nrpairs
    ranks(idx(i)) = i;
%    fprintf('Pair %d has decision %d, rank %d (conf = %e; forward.conf = %e, backward.conf = %e)\n',filteredpairs(idx(i)),decisions(idx(i)),i,confidences(idx(i)),confs_forward(idx(i)),confs_backward(idx(i)));
%    fprintf('Pair %d has rank %d (conf = %e)\n',filteredpairs(idx(i)),i,confidences(idx(i)));
  end

return
