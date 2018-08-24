function [use_this_pair] = filter_onedim_maxpairs (pair, metadata, dataX, dataY, maxpairs)
% function [use_this_pair] = filter_onedim_maxpairs (pair, metadata, dataX, dataY, maxpairs)
%
% Skip pairs with higher-dimensional cause/effect or for which pair > maxpairs
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.
  if metadata(pair,2) ~= metadata(pair,3) || metadata(pair,4) ~= metadata(pair,5) || pair > maxpairs
    use_this_pair = 0;
  else
    use_this_pair = 1;
  end
return
