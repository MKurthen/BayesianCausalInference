function [decisions] = calc_decisions(decs,ranks)
% function [decisions] = calc_decisions(decs,ranks)
%
% Calculates the decisions statistics for visualization
%
% INPUT:
%   decs                  decs is a cell vector (over methods) of cell vectors (over repetitions) of
%                         a vector of decisions (-1 or +1) for all pairs
%   ranks                 decs is a cell vector (over methods) of cell vectors (over repetitions) of
%                         of vector of ranks of pairs according to confidence
%                           (ranks{rep}(1) being the most confident pair, ranks{rep}(end) the least one in repetition rep)
%
% OUTPUT:
%   decisions             cell array of size nrmethods x nrpairs x nrpairs
%                           decisions{m,nrdec,pair} contains a list with length number of 
%                           repetitions of the experiment; for each repetition, that 
%                           slice of decisions{m,nrdec,pair} gives the decision for the 
%                           pair 'pair' and method 'm', where 'nrdec' gives the number of
%                           decisions taken in total
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  % prepare decisions for ROC curve
  nrmethods = length(decs);
  nrpairs = length(decs{1}{1});
  decisions = cell(nrmethods,nrpairs,nrpairs);
  for method=1:nrmethods
    reps = length(decs{method});
    for rep=1:reps
      assert( nrpairs == length(decs{method}{rep}));
    end
    for nrdec = 1:nrpairs
      for pair = 1:nrpairs
        decisions{method,nrdec,pair} = ones(reps,1) * nan;
        for rep=1:reps
          if (ranks{method}{rep}(pair) <= nrdec)
            decisions{method,nrdec,pair}(rep) = decs{method}{rep}(pair);
          end
        end
      end
    end
  end

return
