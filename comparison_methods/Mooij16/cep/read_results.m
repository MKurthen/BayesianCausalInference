function [filteredpairs,weights,totalpairs,nrinfs,decs,confs,ranks,scores,times] = read_results(datadirs,filter,methods,prefix,strict,randomranks,allownans)
% function [filteredpairs,weights,totalpairs,nrinfs,decs,confs,ranks,scores,times] = read_results(datadirs,filter,methods,prefix,strict,randomranks,allownans)
%
% Reads the results of various methods, filters them, and gathers summary statistics
%
% INPUT:
%   datadirs              cell array of directories where the cause-effect-pair data live, one per method
%                           this directory should contain a file 'pairmeta.txt' with the metadata
%                           and then files pair0001.txt, pair0002.txt, ... with the actual data
%                           as ASCII text files (readable with load('-ascii',...))
%                         if this is a string, this datadir is used for all methods
%   filter                optional function that should return a boolean indicating whether a
%                         certain cause-effect-pair should be taken into account for the summary
%                         statistics; if filter == [], no filtering will be done; otherwise, 
%                         filter should be a function handle of the following form:
%                           function [bool] = filter(pair, metadata, dataX, dataY)
%                         example: @filter_one_dim (uses only pairs with one-dimensional variables)
%   methods               cell array containing the relative path names of the methods
%                         if a pathname contains subdirectories with names '1','2','3',...
%                         then these are considered to be independent repetitions
%   prefix                string to add in front of the elements of methods
%                           (e.g., if methods=={'anm','igci'} and prefix=='out/',
%                            results are loaded from the paths 'out/anm' and 'out/igci')
%   strict                if strict == 0, ignore any errors (non-existing files, etc.)
%   randomranks           if randomranks ~= 0, use random ranks instead of confidence ranks
%   allownans             if allownans == 0, nans in decisions/confidences will be replaced by nondecisions
%
% OUTPUT:
%   filteredpairs         cell array of size nrmethods x 1, each entry containing containing the vector of pairs remaining after filtering
%   weights               cell array of size nrmethods x 1, each entry containing containing the vector of containing the weight of each filtered cause-effect pair
%   totalpairs            cell array of size nrmethods x 1, each entry containing containing the number of data pairs in datadir
%   nrinfs                cell array of size nrmethods x 1, each entry containing a vector 
%                           with the number of -Inf confidences for each repetition
%   decs                  decs is a cell vector (over methods) of cell vectors (over repetitions) of
%                         a vector of decisions (-1 or +1) for all pairs
%   confs                 confs is a cell vector (over methods) of cell vectors (over repetitions) of
%                         a vector of confidence of the decisions for each pair
%   ranks                 ranks is a cell vector (over methods) of cell vectors (over repetitions) of
%                         of vector of ranks of pairs according to confidence
%                           (ranks{rep}(1) being the most confident pair, ranks{rep}(end) the least one in repetition rep)
%   scores                scores is a cell vector (over methods) of cell vectors (over repetitions) of
%                         of vector of scores of pairs (the higher the score, the likelier X->Y and the less likely X<-Y)
%   times                 times is a cell vector (over methods) of cell vectors (over repetitions) of
%                         a vector of computation times for all pairs
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  % convert 'datadir' into {'datadir',...,'datadir'} if necessary
  nrmethods = length(methods);
  if isstr(datadirs)
    datadir = datadirs;
    datadirs = cell(nrmethods,1);
    for method=1:nrmethods
      datadirs{method} = datadir;
    end
  end

  % read results
  nrinfs = cell(nrmethods,1);
  decs = cell(nrmethods,1);
  confs = cell(nrmethods,1);
  ranks = cell(nrmethods,1);
  scores = cell(nrmethods,1);
  times = cell(nrmethods,1);
  filteredpairs = cell(nrmethods,1);
  weights = cell(nrmethods,1);
  totalpairs = cell(nrmethods,1);
  for method=1:nrmethods
    % read metadata
    [filteredpairs{method},weights{method},totalpairs{method}] = read_metadata(datadirs{method},filter);
    % nrpairs{method} = length(filteredpairs{method});
    % totalweight = sum(weights{method});

    methoddir = dir(sprintf('%s%s',prefix,methods{method}));
    names = {methoddir.name};
    isdir = [methoddir.isdir];
    reps = 0;
    for entry=1:length(names)
      if isdir(entry) && ~strcmp(names{entry},'.') && ~strcmp(names{entry},'..')
        % check whether the name is an integer (then we have repetitions)
        if strcmp(int2str(str2double(names{entry})),names{entry})
          reps = max([reps, str2double(names{entry})]);
        end
      end
    end
    decs{method} = cell(reps,1);
    confs{method} = cell(reps,1);
    ranks{method} = cell(reps,1);
    scores{method} = cell(reps,1);
    times{method} = cell(reps,1);
    for rep=1:max(reps,1)
      fprintf('Reading method %s, repetition %d...\n', methods{method}, rep);
      if reps > 0
        outdir = sprintf('%s%s/%d',prefix,methods{method},rep);
      else
        outdir = sprintf('%s%s',prefix,methods{method});
      end
      [decs{method}{rep},confs{method}{rep},ranks{method}{rep},scores{method}{rep},times{method}{rep}] = read_decisions(outdir,filteredpairs{method},strict,allownans);
      if randomranks
        ranks{method}{rep} = randperm(length(ranks{method}{rep}));
      end
      nrinf = sum(isinf(confs{method}{rep}) .* (confs{method}{rep} < 0)); % number of -Inf in confidences
      fprintf('Number of -Inf: %d\n',nrinf);

      % number of infinities
      nrinfs{method} = [nrinfs{method}; nrinf];
    end
  end

return
