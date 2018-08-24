function plot_results_bars(datadirs,maxpairs,methods,grouplabels,variantlabels,strict,drawlegend,type)
% function plot_results_bars(datadirs,maxpairs,methods,grouplabels,variantlabels,strict,drawlegend,type)
%
% Reads the results of various methods, filters them, gathers summary statistics and visualises those
% by making box plots of the accuracy, AUC or computation time. 
% Methods are grouped together, and the x-axis will be labeled with the grouplabels. 
% Within each group, methods will be given different colors corresponding to
% the variant; the legend gives for all colors the corresponding variantlabels. 
% The number of methods should be length(grouplabels) * length(variantlabels)
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
%   grouplabels           cell array containing the names of the groups of methods
%   variantlabels         cell array containing the names of the variants of methods within a group
%   strict                if strict == 0, ignore any errors (non-existing files, etc.)
%   drawlegend            if drawlegend ~= 0 and not empty, draw a legend at this position;
%                           ([0,0] means left-bottom, [1,1] means right-top)
%   type                  one of {'p','auc','time'}
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  if( length(grouplabels) * length(variantlabels) ~= length(methods) || length(datadirs) ~= length(methods) )
    error('Invalid arguments');
  end
  nrmethods = length(methods);

  % gather statistics
  total = zeros(nrmethods,1);
  right = zeros(nrmethods,1);
  wrong = zeros(nrmethods,1);
  phat = zeros(nrmethods,1);
  pcil = zeros(nrmethods,1);
  pciu = zeros(nrmethods,1);
  perr = zeros(nrmethods,1);
  pcistdl = zeros(nrmethods,1);
  pcistdu = zeros(nrmethods,1);
  nrinfs = zeros(nrmethods,1);
  auchat = zeros(nrmethods,1);
  auccil = zeros(nrmethods,1);
  aucciu = zeros(nrmethods,1);
  auccistdl = zeros(nrmethods,1);
  auccistdu = zeros(nrmethods,1);
  times = zeros(nrmethods,1);
  for method = 1:nrmethods
    if strcmpi(type,'p') || strcmpi(type,'time')
      stats = calc_results_stats(datadirs{method},maxpairs,{methods{method}},'',strict,0,0,0);
    elseif strcmpi(type,'auc')
      stats = calc_results_stats(datadirs{method},maxpairs,{methods{method}},'',strict,0,0,1000);
    else
      error('unknown type');
    end
    nrpairs = stats.nrpairs;

    % calculate mean scores over reps
    % forced choice decision scenario
    total(method) = mean(stats.total{1,nrpairs});
    right(method) = mean(stats.right{1,nrpairs});
    wrong(method) = mean(stats.wrong{1,nrpairs});
    phat(method) = mean(stats.phat{1,nrpairs});
    pcil(method) = mean(stats.pcil{1,nrpairs});
    pciu(method) = mean(stats.pciu{1,nrpairs});
    perr(method) = mean(stats.perr{1,nrpairs});
    pcistdl(method) = mean(stats.pcistdl{1,nrpairs});
    pcistdu(method) = mean(stats.pcistdu{1,nrpairs});
    nrinfs(method) = stats.nrinf;
    auchat(method) = mean(stats.auchat{1});
    auccil(method) = mean(stats.auccil{1});
    aucciu(method) = mean(stats.aucciu{1});
    auccistdl(method) = mean(stats.auccistdl{1});
    auccistdu(method) = mean(stats.auccistdu{1});
    assert(~isnan(phat(method)));
    assert(~isnan(pcil(method)));
    assert(~isnan(pciu(method)));
    times(method) = mean(stats.times{1});
  end

  % plot bars
  set(gca,'FontSize',8);
  hold on
  co = [0,0,1;0,0.5,0;1,0,0;0,0.75,0.75;0.75,0,0.75;0.75,0.75,0;0.25,0.25,0.25;0,1,0;0,0,0];

  linestyles = {'-','--',':','.'};
  xspace = 8 - length(variantlabels);
  xmax = nrmethods + (length(grouplabels))*xspace + 1;
  xlim([0 xmax]);
  if strcmpi(type,'time')
    ymax = 5.01;
  else
    ymax = 1.01;
  end
  ylim([0 ymax]);

  set(gca,'Units','centimeters');
  set(gca,'Position',[1.5 2 1.5+xmax/10 5]);
  set(gcf,'Units','centimeters');
  set(gcf,'Position',[0 0 1.5+xmax/10+2 1+5+1]);

  % grouping
  if ~strcmpi(type,'time')
    line([0,xmax],[.5,.5],'LineStyle',':','Color','k');
  end
  for method=1:nrmethods
    group = floor((method - 1) / length(variantlabels)) + 1;
    variant = mod(method - 1, length(variantlabels)) + 1;

    % Print and visualize statistics, averaged over repetitions
    if strcmpi(type,'time')
      fprintf('Computation time %s: %f\n', methods{method}, times(method));
    else
      fprintf('Forced-choice accuracy %s: %f+-%f (95%% confidence interval [%f,%f], 68%% confidence interval [%f,%f]), AUC %f (95%% confidence interval [%f,%f], 68%% confidence interval [%f,%f]), total %d right %d wrong %d\n',methods{method},phat(method),perr(method),pcil(method),pciu(method),pcistdl(method),pcistdu(method),auchat(method),auccil(method),aucciu(method),auccistdl(method),auccistdu(method),total(method),right(method),wrong(method));
    end
    white = [1 1 1];

    xpos = (group - 1) * (length(variantlabels) + xspace) + variant;

%    curcol = co(1+mod(method-1,size(co,1)),:);
%    curlst = 1 + mod(floor((method-1) / size(co,1)),length(linestyles));
    curcol = co(variant,:);

    % barplots
    xl = -0.3; xr=0.3; 
    if strcmpi(type,'time')
      cil=0;
      ciu=log(times(method)) / log(10);
      bla = fill([xpos+xl,xpos+xr,xpos+xr,xpos+xl],[cil,cil,ciu,ciu],curcol);
      set(bla,'EdgeColor',curcol);
    else
      if strcmpi(type,'p')
        hat=phat(method); cil=pcil(method); ciu=pciu(method); cistdl=pcistdl(method); cistdu=pcistdu(method);
      elseif strcmpi(type,'auc')
        hat=auchat(method); cil=auccil(method); ciu=aucciu(method); cistdl=auccistdl(method); cistdu=auccistdu(method);
      else
        error('unknown type');
      end
      bla = fill([xpos+xl,xpos+xr,xpos+xr,xpos+xl],[cil,cil,ciu,ciu],curcol * 0.3 + 0.7 * white);
      set(bla,'EdgeColor',curcol * 0.3 + 0.7 * white);
      bla = fill([xpos+xl,xpos+xr,xpos+xr,xpos+xl],[cistdl,cistdl,cistdu,cistdu],curcol * 0.5 + 0.5 * white);
      set(bla,'EdgeColor',curcol * 0.5 + 0.5 * white);
      line([xpos+xl,xpos+xr],[hat,hat],'LineWidth',2,'Color',curcol);
    end
  end
  % grid
  for group = 2:length(grouplabels)
    xpos = (group - 1) * (length(variantlabels) + xspace);
    line([xpos,xpos],[0,ymax],'LineStyle',':','Color','k');
  end
  if ~strcmpi(type,'time')
    line([0,xmax],[0.5,0.5],'LineStyle',':','Color','k');
  end
  set(gca,'XTick',([1:length(grouplabels)] - 1) * (length(variantlabels) + xspace) + length(variantlabels)/2);
%  set(gca,'XTickLabel',grouplabels);

  % Make rotated XTickLabels
  set(gca,'XTickLabel','');
  ax = axis;     % Current axis limits
  axis(axis);    % Set the axis limit modes (e.g. XLimMode) to manual
  Yl = ax(3:4);  % Y-axis limits
  Xt = ([1:length(grouplabels)] - 1) * (length(variantlabels) + xspace) + length(variantlabels)/2;
  t = text(Xt,Yl(1)*ones(1,length(Xt)),grouplabels);
  set(t,'HorizontalAlignment','left','VerticalAlignment','top','Rotation',-45,'FontSize',8);

  if ~strcmpi(type,'time')
    % add nrinfs
    for method=1:nrmethods
      group = floor((method - 1) / length(variantlabels)) + 1;
      variant = mod(method - 1, length(variantlabels)) + 1;
      if( nrinfs(method) >= 1 )
        xpos = (group - 1) * (length(variantlabels) + xspace) + variant;
        if strcmpi(type,'p')
          hat=phat(method);
        elseif strcmpi(type,'auc')
          hat=auchat(method);
        else
          error('unknown type');
        end
        text(xpos,hat,sprintf('%d',nrinfs(method)),'FontSize',6);
      end
    end
  end

  % legend
  xleg = xmax - xspace; %xmax / 25;
  xlegskip = 1; %xmax / 25;
  yleg = ymax * .05;
  ylegskip = ymax * .05;
  for variant = 1:length(variantlabels)
    ypos = yleg + (length(variantlabels) - variant) * ylegskip;
    text(xleg+2*xlegskip,ypos,variantlabels{variant},'FontSize',8);
    curcol = co(variant,:);
    line([xleg,xleg+xlegskip],[ypos,ypos],'LineWidth',2,'Color',curcol);
  end

  if strcmpi(type,'p')
    ylabel('Accuracy');
  elseif strcmpi(type,'auc')
    ylabel('AUC');
  elseif strcmpi(type,'time')
    ylabel('log_{10} of Computation Time (s)');
  else
    error('unknown type');
  end
  hold off

return
