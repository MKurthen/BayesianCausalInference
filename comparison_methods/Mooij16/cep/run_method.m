function result = run_method (pair,method,methodpars,outdir,preprocessing,datadir)
% function result = run_method (pair,method,methodpars,outdir,preprocessing,datadir)
%
% Wrapper that runs a causal inference method on a cause effect pair 
% and writes results to a file in outdir. It also returns its decision 
% about the most likely causal direction and its confidence value. 
%
% Loads variables cause,effect from the cause-effect pair, applies preprocessing and invokes:
%   result = method(cause,effect,methodpars)
% It saves the result in directory outdir in a file named pair0001.hdf5 (or similar)
%
% INPUT:   pair           number of the cause effect pair
%          method         function handle of the method to call (see also cep_template)
%          methodpars     parameters for the method
%          outdir         output directory
%          preprocessing  the preprocessing to apply to the data (see also load_pair)
%          datadir        data directory containing the pairs
%
% OUTPUT:  result         result struct returned by cep_* (see also cep_template.m)
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  % print diagnostics
  fprintf('Running method %s on pair %d in %s\n',method,pair,datadir);
  methodpars,
  preprocessing,

  % abort if destination file exists already
  outfile = sprintf('%s/pair%04d.hdf5',outdir,pair);
  if exist(outfile,'file')
    % skip if file already exists
    fprintf('%s already exists! Loading existing results...\n', outfile);
    result = load('-mat',outfile);
    return;
  end;

  % load data and apply preprocessing
  [cause,effect] = load_pair(pair,preprocessing,datadir);

  % create outdir
  [success,msg] = mkdir(outdir);

  % run causal inference method
  result = feval(method,cause,effect,methodpars);

  % write results to file
  save(outfile,'result','-v7.3');

return
