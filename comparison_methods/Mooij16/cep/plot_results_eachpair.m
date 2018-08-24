function plot_results_eachpair(datadir,maxpairs,methods,prefix,decrate,labels,strict)
% function plot_results_eachpair(datadir,maxpairs,methods,prefix,decrate,labels,strict)
%
% Reads the results of various methods, filters them, gathers summary statistics and visualises those
% as a heat map showing the decisions for each pair and each repetition individually.
%
% INPUT:
%   datadir               directory where the cause-effect-pair data live
%                           this directory should contain a file 'pairmeta.txt' with the metadata
%                           and then files pair0001.txt, pair0002.txt, ... with the actual data
%                           as ASCII text files (readable with load('-ascii',...))
%   maxpairs              maximum pair id to consider (use Inf if no maximum is required)
%   methods               cell array containing the relative path names of the methods
%   prefix                string to add in front of the elements of methods
%                           (e.g., if methods=={'anm','igci'} and prefix=='out/',
%                            results are loaded from the paths 'out/anm' and 'out/igci')
%   decrate               decision rate (fraction of decisions to take)
%   labels                cell array containing the names of the methods to use in the legend
%                           (if []: same as methods)
%   strict                if strict == 0, ignore any errors (non-existing files, etc.)
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  if isempty(labels)
    labels = methods;
  end

  % read results
  filter = @(pair,metadata,dataX,dataY) filter_onedim_maxpairs (pair, metadata, dataX, dataY, maxpairs);
  [filteredpairs,weights,totalpairs,nrinfs,decs,confs,ranks,scores] = read_results(datadir,filter,methods,prefix,strict,0,1);
  filteredpairs = filteredpairs{1};
  weights = weights{1};
  totalpairs = totalpairs{1};
  decisions = calc_decisions(decs, ranks);

  nrpairs = length(filteredpairs);
  totalweight = sum(weights);
  nrmethods = length(methods);

  % calculate weights for all pairs (pairs that are not included after filtering get weight 0)
  wgt = zeros(totalpairs,1);
  for fpair=1:nrpairs
    wgt(filteredpairs(fpair)) = weights(fpair);
  end

  % count number of reps for each method
  reps = ones(nrmethods,1);
  for method=1:nrmethods
    reps(method) = length(decisions{method,nrpairs,1});
  end

  % take decisions at requested decision rate and fill in filtered pairs
  % for each method, all repetitions are combined into a long vector
  nrdecs = ceil(decrate * nrpairs);
  dec = cell(nrmethods,1);
  wgts = cell(nrmethods,1);
  for method=1:nrmethods
    dec{method} = ones(1,totalpairs * reps(method)) * 1.5;
    wgts{method} = zeros(1,totalpairs * reps(method));
    for fpair=1:nrpairs
      dec{method}(((filteredpairs(fpair)-1) * reps(method) + 1):(filteredpairs(fpair) * reps(method))) = decisions{method,nrdecs,fpair};
      wgts{method}(((filteredpairs(fpair)-1) * reps(method) + 1):(filteredpairs(fpair) * reps(method))) = ones(1,reps(method)) * wgt(filteredpairs(fpair));
    end
  end

  % plot decisions
  figure;
  ax1 = subplot('Position',[0.2 0.1 0.65 0.8]);
  hold on
  for method=1:nrmethods
    imagesc([0.5 / reps(method), totalpairs - 0.5 / reps(method)],[method],dec{method});
    % calculate accuracies in forced-decision scenario
    total = sum(ones(size(dec{method})) .* wgts{method});
    right = sum((dec{method} == 1) .* wgts{method});
    wrong = sum((dec{method} == -1) .* wgts{method});
    fprintf('%s: total %3.1f, right %3.1f, wrong %3.1f, undecided %3.1f, accuracy %3.1f\n',labels{method},total,right,wrong,total-right-wrong,right/(total)*100);
    t=text(totalpairs+1,method,sprintf('%3.1f%%',binofit(round(right),ceil(total))*100));
  end
  hold off
  set(ax1,'YTick',[1:nrmethods]);
  set(ax1,'YTickLabel',labels);
  colormap([[linspace(1,0,100) linspace(0,0,100) linspace(0,1,100)];[linspace(0,0,100) linspace(0,1,100) linspace(0,1,100)];[linspace(0,0,100) linspace(0,0,100) linspace(0,1,100)]]');
  caxis(ax1,[-1,2.001]);
  xlim([0,totalpairs]);
  xlabel(ax1,'Cause-Effect Pair #');

  % plot weights
  ax2 = subplot('Position',[0.2 0.925 0.65 0.05]);
  imagesc(wgt'+1.001);
  caxis(ax2,[-1,2.001]);
  set(ax2,'XTick',[]);
  set(ax2,'XTickLabel','');
  set(ax2,'YTick',[1]);
  set(ax2,'YTickLabel','Weights');
  xlabel('');
  ylabel('');

return
