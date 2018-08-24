function plot_results_roccurves(datadirs,maxpairs,methods,labels,strict,drawlegend,randomlabels,linecolors,linestyles,linewidths)
% function plot_results_roccurves(datadirs,maxpairs,methods,labels,strict,drawlegend,randomlabels,linecolors,linestyles,linewidths)
%
% Reads the results of various methods, filters them, gathers summary statistics and visualises those
% as a ROC curve.
%
% NOTE: the user should specify a datadir for each method!
%
% INPUT:
%   datadirs              cell array containing the directories where the cause-effect-pair data live
%                           each directory should contain a file 'pairmeta.txt' with the metadata
%                           and then files pair0001.txt, pair0002.txt, ... with the actual data
%                           as ASCII text files (readable with load('-ascii',...))
%   maxpairs              maximum pair id to consider (use Inf if no maximum is required)
%   methods               cell array containing the relative path names of the methods
%   labels                cell array containing the names of the methods
%   strict                if strict == 0, ignore any errors (non-existing files, etc.)
%   drawlegend            if drawlegend ~= 0 and not empty, draw a legend at this position;
%                           ([0,0] means left-bottom, [1,1] means right-top)
%   randomlabels          if randomlabels ~= 0, randomize "true" causal directions
%   linecolors            nrmethods x 3 array containing RGB triples of colors to be used for the curves
%                           (e.g., [0,0,1; 1,0,0] results in a blue and a red curve)
%   linestyles            cell vector containing linestyles to be used for the curves
%                           (e.g., {'-','.'} results in a solid and a dotted curve)
%   linewidths            cell vector containing linewidths to be used for the curves
%                           (e.g., {0.5,1.0} results in a standard and double width curve)
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  if( length(labels) ~= length(methods) || length(datadirs) ~= length(methods) )
    error('Invalid arguments');
  end
  nrmethods = length(methods);
  filter = @(pair,metadata,dataX,dataY) filter_onedim_maxpairs (pair, metadata, dataX, dataY, maxpairs);
  % read results
  [filteredpairs,weights,totalpairs,nrinfs,decs,confs,ranks,scores] = read_results(datadirs,filter,methods,'',strict,0,0);

  % plot curves
  set(gca,'FontSize',10);
  hold on
  line([0,1],[0,1],'LineStyle',':','Color','k');

  p = [];
  AUCs = zeros(nrmethods,1);
  for method=1:nrmethods
%   % read results
%   [filteredpairs,weights,totalpairs,nrinfs,decs,confs,ranks] = read_results(datadirs{method},filter,{methods{method}},'',strict,0,0);

    for rep=1:length(decs{method})
      % why do we need to do this???
      if randomlabels % random labels
        truelabels = (rand(size(confs{method}{rep})) < 0.5) * 2 - 1;
      else
        truelabels = mod([1:length(confs{method}{rep})]',2) * 2 - 1;
      end

      %thescores = (decs{method}{rep} .* (-tanh((ranks{method}{rep} - totalpairs{method}) / totalpairs{method})+1)/2);
      thescores = scores{method}{rep};
      [X,Y,T,AUC] = perfcurve(truelabels,truelabels .* thescores,1,'Weights',weights{method});
      %X = X(:,1); % not interested in CI
      %Y = Y(:,1); % not interested in CI
      % AUC = [hi lo mid]

      % curcol = co(1+mod(method-1,size(co,1)),:);
      curcol = linecolors(method,:);
      % curlst = 1 + mod(floor((method-1) / size(co,1)),length(linestyles));
      curlst = linestyles{method};
      curlwd = linewidths{method};
      p(method) = plot(X,Y,curlst,'Color',curcol,'LineWidth',curlwd);

      AUCs(method) = AUC;
    end
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
  xlabel('FPR','FontSize',8);
  ylabel('TPR','FontSize',8);
  hold off

%  for method=1:nrmethods
%    fprintf('AUC %s = %e CI [%e,%e]\n',methods{method},AUCs(method,1),AUCs(method,2),AUCs(method,3));
%  end

return
