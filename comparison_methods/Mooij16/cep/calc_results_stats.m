function stats = calc_results_stats(datadir,maxpairs,methods,prefix,strict,randomranks,allownans,aucci)
% function stats = calc_results_stats(datadir,maxpairs,methods,prefix,strict,randomranks,allownans,aucci)
%
% Reads the results of various methods, filters them, and calculates summary statistics.
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
%   strict                if strict == 0, ignore any errors (non-existing files, etc.)
%   randomranks           if randomranks ~= 0, use random rankings instead of confidence rankings
%   allownans             if allownans == 0, nans in decisions/confidences will be replaced by nondecisions
%   aucci                 if aucci ~= 0, calculate AUC confidence intervals by bootstrapping (slow!)
%                           using aucci bootstrap samples
%
% OUTPUT:
%   stats:                struct with fields:
%     totalweight              sum of the weights of the pairs
%     nrpairs                  number of pairs after filtering
%     nrinf                    number of pairs with confidence -infinity or NaN
%     right{method,nrdec}      reps x 1 vector (weighted number of correct decisions)
%     wrong{method,nrdec}      reps x 1 vector (weighted number of incorrect decisions)
%     undecided{method,nrdec}  reps x 1 vector (weighted number of nondecisions)
%     total{method,nrdec}      reps x 1 vector (weighted number of total decisions)
%     phat{method,nrdec}       reps x 1 vector (estimate of success probability p)
%     pcil{method,nrdec}       reps x 1 vector (lower 95% confidence interval for p)
%     pciu{method,nrdec}       reps x 1 vector (upper 95% confidence interval for p)
%     perr{method,nrdec}       reps x 1 vector (standard deviation of success probability p estimate)
%     pcistdl{method,nrdec}    reps x 1 vector (lower 68% confidence interval for p)
%     pcistdu{method,nrdec}    reps x 1 vector (upper 68% confidence interval for p)
%     auchat{method}           reps x 1 vector (estimate of AUC)
%     auccil{method}           reps x 1 vector (lower 95% confidence interval for AUC)
%     aucciu{method}           reps x 1 vector (upper 95% confidence interval for AUC)
%     auccistdl{method}        reps x 1 vector (lower 68% confidence interval for AUC)
%     auccistdu{method}        reps x 1 vector (upper 68% confidence interval for AUC)
%     times{method}            reps x 1 vector (total computation time per method)
%                         here reps is the number of repetitions of that method
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  % read results
  filter = @(pair,metadata,dataX,dataY) filter_onedim_maxpairs (pair, metadata, dataX, dataY, maxpairs);
  [filteredpairs,weights,totalpairs,nrinfs,decs,confs,ranks,scores,times] = read_results(datadir,filter,methods,prefix,strict,randomranks,allownans);
  filteredpairs = filteredpairs{1};
  decisions = calc_decisions(decs, ranks);

  nrpairs = length(filteredpairs);
  totalweight = sum(weights{1});

  nrmethods = length(methods);

  % calculate statistics
  alpha = 1 - (normcdf(1,0,1) - normcdf(-1,0,1));  % confidence interval corresponding to about 2 standard deviations
  total = cell(nrmethods,nrpairs);
  right = cell(nrmethods,nrpairs);
  wrong = cell(nrmethods,nrpairs);
  undecided = cell(nrmethods,nrpairs);
  phat = cell(nrmethods,nrpairs);
  rcil = cell(nrmethods,nrpairs);
  pciu = cell(nrmethods,nrpairs);
  perr = cell(nrmethods,nrpairs);
  pcistdl = cell(nrmethods,nrpairs);
  pcistdu = cell(nrmethods,nrpairs);
  auchat = cell(nrmethods,1);
  auccil = cell(nrmethods,1);
  aucciu = cell(nrmethods,1);
  auccistdl = cell(nrmethods,1);
  auccistdu = cell(nrmethods,1);
  totaltimes = cell(nrmethods,1);
  for method = 1:nrmethods
    for nrdec = 1:nrpairs
      % number of repetitions
      reps = length(decisions{method,nrdec,1});
      % reps x nrpairs decisions
      decis = [decisions{method,nrdec,:}];
      for rep = 1:reps
        dec = decis(rep,:);
        right{method,nrdec} = [right{method,nrdec}; (dec == 1) * weights{1}];
        wrong{method,nrdec} = [wrong{method,nrdec}; (dec == -1) * weights{1}];
        undecided{method,nrdec} = [undecided{method,nrdec}; (dec == 0) * weights{1}];
      end
      total{method,nrdec} = right{method,nrdec} + wrong{method,nrdec} + undecided{method,nrdec};

      % estimate success probability and 95% confidence intervals of Bernoulli experiment
      [phats, pcis] = binofit(round(right{method,nrdec}),ceil(total{method,nrdec}));
      phat{method,nrdec} = phats;
      pcil{method,nrdec} = pcis(:,1);
      pciu{method,nrdec} = pcis(:,2);

      % estimate success probability and 68% confidence intervals of Bernoulli experiment
      [dummy, pcisstd] = binofit(round(right{method,nrdec}),ceil(total{method,nrdec}),alpha);
      perr{method,nrdec} = (pcisstd(:,2) - pcisstd(:,1)) / 2;
      pcistdl{method,nrdec} = pcisstd(:,1);
      pcistdu{method,nrdec} = pcisstd(:,2);
    end

    % estimate AUC
    auchat{method} = zeros(reps,1);
    auccil{method} = zeros(reps,1);
    aucciu{method} = zeros(reps,1);
    auccistdl{method} = zeros(reps,1);
    auccistdu{method} = zeros(reps,1);
    for rep=1:reps
      randomlabels = 1;
      if randomlabels
        truelabels = (rand(size(confs{method}{rep})) < 0.5) * 2 - 1;
      else
        truelabels = mod([1:length(confs{method}{rep})]',2) * 2 - 1;
      end
      %thescores = (decs{method}{rep} .* (-tanh((ranks{method}{rep} - totalpairs{method}) / totalpairs{method})+1)/2);
      thescores = scores{method}{rep};
      if ~aucci
        [dumX,dumY,dumT,AUC] = perfcurve(truelabels,truelabels .* thescores,1,'Weights',weights{method}); %,'NBoot',1000);
        auchat{method}(rep) = AUC;
        auccil{method}(rep) = nan;
        aucciu{method}(rep) = nan;
        auccistdl{method}(rep) = nan;
        auccistdu{method}(rep) = nan;
      else
        [dumX,dumY,dumT,AUC] = perfcurve(truelabels,truelabels .* thescores,1,'Weights',weights{method},'NBoot',aucci,'BootType','per');
        % the default BootType ('bca') and 'cper' sometimes yield weird results: the estimated AUC can be outside its 95% confidence interval
        % therefore we decided not to use bias correction (the MatLab documentation isn't so helpful as to provide pointers to descriptions 
        % of these methods...)
        auchat{method}(rep) = AUC(1);
        auccil{method}(rep) = AUC(2);
        aucciu{method}(rep) = AUC(3);
        [dumX,dumY,dumT,AUC] = perfcurve(truelabels,truelabels .* thescores,1,'Weights',weights{method},'NBoot',aucci,'BootType','per','Alpha',alpha);
        auccistdl{method}(rep) = AUC(2);
        auccistdu{method}(rep) = AUC(3);
      end
    end

    % calculate total computation time
    totaltimes{method} = zeros(reps,1);
    for rep=1:reps
      totaltimes{method}(rep) = sum(times{method}{rep});
    end
  end

  stats = struct;
  stats.totalweight = totalweight;
  stats.nrpairs = nrpairs;
  stats.nrinf = max([nrinfs{:}]);
  stats.total = total;
  stats.right = right;
  stats.wrong = wrong;
  stats.undecided = undecided;
  stats.phat  = phat;
  stats.pcil  = pcil;
  stats.pciu  = pciu;
  stats.perr  = perr;
  stats.pcistdl = pcistdl;
  stats.pcistdu = pcistdu;
  stats.auchat = auchat;
  stats.auccil = auccil;
  stats.aucciu = aucciu;
  stats.auccistdl = auccistdl;
  stats.auccistdu = auccistdu;
  stats.times = totaltimes;

return
