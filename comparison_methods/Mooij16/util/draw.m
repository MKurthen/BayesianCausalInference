function [k] = draw(m,N,p)
% function [k] = draw(m,N,p)
%
%   Draws m samples out of N (without replacement) with probability 
%   distribution p. If p is omitted, it is assumed to be uniform
%   (in which case the algorithm speeds up a lot!).
%   Returns a vector of m integers in the range 1..N
%
% Copyright (c) 2011  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

if nargin==2
  % this special case is simple: generate a random permutation
  % using randperm and select the first m numbers
  k = randperm(N);
  k = k(1:m);
else
  if length(p) ~= N
    error('p should be vector of length N');
  end

  for j=1:m
    Z = sum(p);
    if( Z == 0 )
      m
      N
      p
      j
      error('draw: Z == 0');
    end;
    p = p / Z;  % normalize
    [q,b] = sort(p); % sort for speed improvement

    r = rand(1); % throw die
    qsum = 0.0;
    i = 0;
    while (qsum < r) && (i < N) % go from highest-prob to lowest-prob because of speed
      qsum = qsum + q(N-i);
      i = i + 1;
    end
    if i == 0
      error('draw: i=0, should not happen');
    end;
    i = b(N-i+1); % draw this one
    
    k(j) = i;
    p(i) = 0;
  end    
end

return;
