function [filteredpairs,weights,totalpairs] = read_metadata(datadir,filter)
% function [filteredpairs,weights,totalpairs] = read_metadata(datadir,filter)
%
% Reads the metadata from datadir and filters the cause-effect pairs in that directory according to the chosen filter
%
% INPUT:
%   datadir               directory where the cause-effect-pair data live (e.g., '../../webdav')
%                           this directory should contain a file 'pairmeta.txt' with the metadata
%                           and files pair0001.txt, pair0002.txt, ... with the actual data
%                           as ASCII text files (readable with load('-ascii',...))
%   filter                optional function that should return a boolean indicating whether a
%                         certain cause-effect-pair should be taken into account for the summary
%                         statistics; if filter == [], no filtering will be done; otherwise, 
%                         filter should be a function handle of the following form:
%                           function [bool] = filter(pair, metadata, dataX, dataY)
%                         example: @filter_one_dim (retains only pairs with one-dimensional variables)
% OUTPUT:
%   filteredpairs         vector of cause-effect pairs remaining after filtering
%   weights               vector containing the weight of each filtered cause-effect pair
%   totalpairs            total number of (unfiltered) cause-effect pairs
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  % load metadata
  metadata = load(sprintf('%s/pairmeta.txt',datadir));
  totalpairs = size(metadata,1);

  % consistency check
  for pair=1:totalpairs
    if metadata(pair,1) ~= pair 
      error(sprintf('Invalid metadata in %s/pairmeta.txt',datadir));
    end
  end

  % set weights
  weights = metadata(1:totalpairs,6);

  % filter pairs
  usepair = ones(totalpairs,1);
  if ~isempty(filter)
    for pair=1:totalpairs
      % load data
      filename = sprintf('%s/pair%04d.txt',datadir,pair);
      data = load('-ascii',filename);
      % assign cause and effect
      dataX = data(:,metadata(pair,2):metadata(pair,3));
      dataY = data(:,metadata(pair,4):metadata(pair,5));

      % call filtering function
      usepair(pair) = feval(filter, pair, metadata, dataX, dataY);
    end
  end

  % filteredpairs are the remaining pairs after filtering
  filteredpairs = find(usepair);

  % weights are their weights
  weights = weights(filteredpairs);

return
