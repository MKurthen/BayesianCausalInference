function perm = myrandperm(n,seed)
% function perm = myrandperm(n,seed)
%
% Returns a random permutation of the integers from 1 to n. 
%
% INPUT:  n        number of elements
%         seed     random number seed (integer; default: 12345)
%
% Output: perm     random permutation of {1,2,...,n}
%
% Copyright (c) 2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  if nargin < 2
    seed = 12345;
  end
  s = RandStream('mcg16807','Seed',seed);
  perm = randperm(s,n);

return
