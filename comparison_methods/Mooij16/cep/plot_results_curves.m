function plot_results_curves(datadir,maxpairs,methods,prefix,type,labels,strict,drawlegend)
% function plot_results_curves(datadir,maxpairs,methods,prefix,type,labels,strict,drawlegend)
%
% Reads the results of various methods, filters them, gathers summary statistics and visualises those
% as an decision rate vs. accuracy plot. If a method has more than one repetition, the 'curves' type
% will plot the curves corresponding to each repetition and an average curve, and the 'shapes' type
% will plot averaged shapes.
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
%   type                  one of {'curves','shapes'}
%   labels                cell array containing the names of the methods to use in the legend
%                           (if []: same as methods)
%   strict                if strict == 0, ignore any errors (non-existing files, etc.)
%   drawlegend            if drawlegend ~= 0 and not empty, draw a legend at this position;
%                           ([0,0] means left-bottom, [1,1] means right-top)
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  if isempty(labels)
    labels = methods;
  end
  if ~strcmp(type,'curves') && ~strcmp(type,'shapes')
    error('Unknown type');
  end

  % gather statistics
  stats = calc_results_stats(datadir,maxpairs,methods,prefix,strict,0,0,0);
  totalweight = stats.totalweight;
  total = stats.total;
  right = stats.right;
  wrong = stats.wrong;
  phat = stats.phat;
  pcil = stats.pcil;
  pciu = stats.pciu;
  perr = stats.perr;
  pcistdl = stats.pcistdl;
  pcistdu = stats.pcistdu;
  nrmethods = length(methods);
  nrpairs = stats.nrpairs;

  % plot curves
%  figure;
  xlim([0 100]);
  ylim([0 101]);
  set(gca,'FontSize',10);
  hold on
  co = get(gca,'ColorOrder');
  linestyles = {'-','--',':','.'};

  if strcmpi(type,'curves')
    itw = round(totalweight);
    X = 100 * [1:itw] / itw;
    Y = 100 * binoinv(0.975*ones(1,itw),[1:itw],0.5*ones(1,itw)) ./ [1:itw];
    bla = fill([X';flipud(X')],[Y';flipud(100-Y')],[0.5 0.5 0.5]);
    set(bla,'EdgeColor',[0.3 0.3 0.3]);
    h = text(95,90,'Significant');
    set(h,'HorizontalAlignment','right');
    h = text(95,55,'Not significant');
    set(h,'HorizontalAlignment','right');
  end
  plot([0 100],[50 50],'k:');

  p = [];
  for method=1:nrmethods
    totals = [total{method,:}];
    rights = [right{method,:}];
    phats = [phat{method,:}];
    pcils = [pcil{method,:}];
    pcius = [pciu{method,:}];
    pcistdls = [pcistdl{method,:}];
    pcistdus = [pcistdu{method,:}];
    decrates = totals / totalweight;
    assert(sum(sum(isnan(decrates))) == 0);
    assert(sum(sum(isnan(phats))) == 0);
    assert(sum(sum(isnan(pcils))) == 0);
    assert(sum(sum(isnan(pcius))) == 0);

    % Print and visualize statistics, averaged over repetitions
    fprintf('Forced-choice accuracy: %f+-%f (95%% confidence interval [%f,%f], 68%% confidence interval [%f,%f]) sum-of-weights: %f\n',mean(phat{method,nrpairs}),mean(perr{method,nrpairs}),mean(pcil{method,nrpairs}),mean(pciu{method,nrpairs}),mean(pcistdl{method,nrpairs}),mean(pcistdu{method,nrpairs}),totalweight);
    curcol = co(1+mod(method-1,size(co,1)),:);
    curlst = 1 + mod(floor((method-1) / size(co,1)),length(linestyles));
    if strcmpi(type,'shapes')
      white = [1 1 1];
      % 95% confidence interval
      bla = fill([100*mean(decrates,1)';flipud(100*mean(decrates,1)')],[100*mean(pcils,1)';flipud(100*mean(pcius,1)')],curcol * 0.3 + 0.7 * white);
      set(bla,'EdgeColor',curcol * 0.5 + 0.5 * white);
      
      % 68% confidence interval
      bla = fill([100*mean(decrates,1)';flipud(100*mean(decrates,1)')],[100*mean(pcistdls,1)';flipud(100*mean(pcistdus,1)')],curcol * 0.5 + 0.5 * white);
      set(bla,'EdgeColor',curcol * 0.7 + 0.3 * [1 1 1]);
    end
    p(method) = plot(100*mean(decrates,1),100*mean(phats,1),linestyles{curlst},'Color',curcol,'LineWidth',2);
  end

  if( length(drawlegend) == 2 )
    %h_legend = legend(p,labels,'NorthEast');
    h_legend = legend(p,labels,'Orientation','Vertical');
    set(h_legend,'FontSize',8);

    % thanks to Eng. Ahmed Badr for sharing this solution on how to draw a common legend for various subplots
    p_legend = get(h_legend,'Position');
    p_legend(1) = drawlegend(1);
    p_legend(2) = drawlegend(2);
    set(h_legend,'Position',p_legend);
  end
%  xlabel('Decision rate (%)','FontSize',14);
%  ylabel('Accuracy (%)','FontSize',14);
  xlabel('Decision rate (%)','FontSize',8);
  ylabel('Accuracy (%)','FontSize',8);
  hold off

return
