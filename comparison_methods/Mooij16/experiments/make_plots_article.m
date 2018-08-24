function make_plots_article(strict)
% function make_plots_article(strict)
%
% This function produces the plots in [1]. runmethods_article.sh needs to be run first.
%
% INPUT:
%   strict:  boolean that decides what to do with missing files / other problems
%            (ignore if 0, complain otherwise)
%
% [1]  J. M. Mooij, J. Peters, D. Janzing, J. Zscheischler, B. Schoelkopf
%      Distinguishing cause from effect using observational data: methods and benchmarks
%      arXiv:1412.3773v2, submitted to Journal of Machine Learning Research
%      http://arxiv.org/abs/1412.3773v2
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  plotsdir = 'plots_article';
  [success,msg] = mkdir(plotsdir);
  fm = @(c) cellfun(@(s) strrep(s,'_',' '),c,'UniformOutput',0);

  datadirs = {'CEP','SIM','SIM-c','SIM-ln','SIM-G'};
  datadirlabels = {'CEP','SIM','SIM-c','SIM-ln','SIM-G'};
  types = {'p','auc','time'};

  % 'Shannon_spacing_LL' : not enough image toolbox users
  warning('The following may segfault in older MatLab releases...it seems to work in R2014b');
  if 1 % allpairs [IN ARTICLE]
    for i=1:length(datadirs)
      filter = @(pair,metadata,dataX,dataY) filter_onedim_maxpairs (pair, metadata, dataX, dataY, Inf);
      [filteredpairs,weights,totalpairs] = read_metadata(datadirs{i},filter);
      nrpairs = length(filteredpairs);
      totalweight = sum(weights);

      plot_all_pairs(datadirs{i},sprintf('%s/allpairs_%s.pdf',plotsdir,strrep(datadirs{i},'/','_')),filteredpairs,'pdf');
    end
  end

  if 1 % all ANM methods in a single figure, different datadirs but no disturbances [IN ARTICLE]
    methods = {'pHSIC_', 'HSIC_', 'HSIC__split', 'HSIC__fk', 'HSIC__split_fk', 'entropy_KSG', 'entropy_Shannon_spacing_V', 'entropy_Shannon_KDP', 'entropy_Shannon_PSD_SzegoT', 'entropy_Shannon_kNN_k', 'entropy_Shannon_Edgeworth', 'FN_', 'Gauss_', 'MML_', 'entropy_Shannon_MaxEnt1', 'entropy_Shannon_MaxEnt2'}; 
    % Shannon_expF = Gauss
    plot_methods(methods,translate(methods,'ANM-'),datadirs,datadirlabels,'_out/anm_N=Inf_FITC=100_','_lbfgsb',strict,[plotsdir '/ANM_all_N=Inf_FITC=100_lbfgsb'],['unperturbed'],types);
  end
  if 0 % [NOT IN ARTICLE]
    % only *HSIC in a single figure, different datadirs but no disturbances
    methods = {'pHSIC_','HSIC_','HSIC__split','HSIC__fk','HSIC__split_fk'};
    plot_methods(methods,translate(methods,'ANM-'),datadirs,datadirlabels,'_out/anm_N=Inf_FITC=100_','_lbfgsb',strict,[plotsdir '/ANM_allHSIC_N=Inf_FITC=100_lbfgsb'],['unperturbed, ANM-*'],types);
  end
  if 1 % all ANM methods in a single figure, for each datadir separately, with disturbances  [IN ARTICLE FOR datadir=CEP]
    methods = {'pHSIC_', 'HSIC_', 'HSIC__split', 'HSIC__fk', 'HSIC__split_fk', 'entropy_KSG', 'entropy_Shannon_spacing_V', 'entropy_Shannon_KDP', 'entropy_Shannon_PSD_SzegoT', 'entropy_Shannon_kNN_k', 'entropy_Shannon_Edgeworth', 'FN_', 'Gauss_', 'MML_', 'entropy_Shannon_MaxEnt1', 'entropy_Shannon_MaxEnt2'}; 
    % Shannon_expF = Gauss
    methodlabels = translate(methods,'ANM-');
    variants = {'','_disc','_undisc','_dist=1e-9'};
    variantlabels = {'unperturbed','discretized','undiscretized','small noise'};
    for datadir=1:length(datadirs)
      % disturbances
      plot_methods_perturbations(methods,methodlabels,variants,variantlabels,datadirs{datadir},'_out/anm_N=Inf_FITC=100_','_lbfgsb','',strict,sprintf('%s/ANM_all_N=Inf_FITC=100_lbfgsb_%s',plotsdir,datadirlabels{datadir}),sprintf('%s',datadirlabels{datadir}),types);
    end
  end
  if 0 % only *HSIC in a single figure, for each datadir separately, with disturbances  [NOT IN ARTICLE]
    methods = {'pHSIC_','HSIC_','HSIC__split','HSIC__fk','HSIC__split_fk'};
    grouplabels = translate(methods,'ANM-');
    variants = {'','_disc','_undisc','_dist=1e-9'};
    variantlabels = {'unperturbed','discretized','undiscretized','small noise'};
    for datadir=1:length(datadirs)
      plot_methods_perturbations(methods,grouplabels,variants,variantlabels,datadirs{datadir},'_out/anm_N=Inf_FITC=100_','_lbfgsb','',strict,sprintf('%s/ANM_allHSIC_N=Inf_FITC=100_lbfgsb_%s',plotsdir,datadirlabels{datadir}),sprintf('%s',datadirlabels{datadir}),types);
    end
  end
  if 1 % all IGCI methods in a single figure, different datadirs but no disturbances  [IN ARTICLE]
    for thenorm=1:2
      if thenorm == 1
        methods = {'slope_','slope++_','entropy_KSG','entropy_Shannon_spacing_V','entropy_Shannon_KDP','entropy_Shannon_PSD_SzegoT','entropy_Shannon_kNN_k','entropy_Shannon_Edgeworth','entropy_Shannon_MaxEnt1','entropy_Shannon_MaxEnt2','entropy_Shannon_expF'}; 
        % org_entropy_: same as entropy_KSG
        normlabel = 'uniform';
      else
        methods = {'slope_','slope++_','entropy_KSG','entropy_Shannon_spacing_V','entropy_Shannon_KDP','entropy_Shannon_PSD_SzegoT','entropy_Shannon_kNN_k','entropy_Shannon_Edgeworth','entropy_Shannon_MaxEnt1','entropy_Shannon_MaxEnt2'}; 
        % org_entropy_: same as entropy_KSG
        % entropy_Shannon_expF = Gauss
        normlabel = 'Gaussian';
      end
      plot_methods(methods,translate(methods,'IGCI-'),datadirs,datadirlabels,'_out/igci_N=Inf_',sprintf('_norm%d', thenorm),strict,sprintf('%s/IGCI_all_N=Inf_%s',plotsdir,normlabel),sprintf('unperturbed, %s base measure',normlabel),types);
    end
  end
  if 1 % all IGCI methods in a single figure, for each datadir separately: with disturbances  [IN ARTICLE FOR datadir=CEP]
    variants = {'','_disc','_undisc','_dist=1e-9'};
    variantlabels = {'unperturbed','discretized','undiscretized','small noise'};
    for datadir=1:length(datadirs)
      for thenorm=1:2
        if thenorm==1
          methods = {'slope_','slope++_','entropy_KSG','entropy_Shannon_spacing_V','entropy_Shannon_KDP','entropy_Shannon_PSD_SzegoT','entropy_Shannon_kNN_k','entropy_Shannon_Edgeworth','entropy_Shannon_MaxEnt1','entropy_Shannon_MaxEnt2','entropy_Shannon_expF'}; 
          % org_entropy_: same as entropy_KSG
          normlabel = 'uniform';
        else
          methods = {'slope_','slope++_','entropy_KSG','entropy_Shannon_spacing_V','entropy_Shannon_KDP','entropy_Shannon_PSD_SzegoT','entropy_Shannon_kNN_k','entropy_Shannon_Edgeworth','entropy_Shannon_MaxEnt1','entropy_Shannon_MaxEnt2'}; 
          % org_entropy_: same as entropy_KSG
          % entropy_Shannon_expF = Gauss
          normlabel = 'Gaussian';
        end
        % disturbances
        plot_methods_perturbations(methods,translate(methods,'IGCI-'),variants,variantlabels,datadirs{datadir},'_out/igci_N=Inf_',sprintf('_norm%d', thenorm),'',strict,sprintf('%s/IGCI_all_N=Inf_%s_%s',plotsdir,normlabel,datadirlabels{datadir}),sprintf('%s, %s base measure',datadirlabels{datadir},normlabel),types);
      end
    end
  end

  if 0 % all ANM methods in a single figure, for each datadir separately, eachpair [NOT IN ARTICLE, STILL USEFUL]
    methods = {'pHSIC_', 'HSIC_', 'HSIC__split', 'HSIC__fk', 'HSIC__split_fk', 'entropy_KSG', 'entropy_Shannon_spacing_V', 'entropy_Shannon_spacing_Vb', 'entropy_Shannon_spacing_Vpconst', 'entropy_Shannon_spacing_Vplin', 'entropy_Shannon_spacing_Vplin2', 'entropy_Shannon_spacing_VKDE', 'entropy_Shannon_KDP', 'entropy_Shannon_PSD_SzegoT', 'entropy_Shannon_kNN_k', 'entropy_Shannon_Edgeworth', 'FN_', 'Gauss_', 'MML_', 'entropy_Shannon_MaxEnt1', 'entropy_Shannon_MaxEnt2'}; 
    % Shannon_expF = Gauss
    methodlabels = translate(methods,'ANM-');
    fullmethods = {};
    for i=1:length(methods)
      fullmethods{i} = sprintf('anm_N=Inf_FITC=100_%s_lbfgsb',methods{i});
    end
    for datadir=1:length(datadirs)
      plot_results_eachpair(datadirs{datadir},Inf,fullmethods,sprintf('%s_out/',datadirs{datadir}),1,methodlabels,strict);
      print('-depsc2',sprintf('%s/ANM_eachpair_N=Inf_FITC=100_lbfgsb_%s.eps',plotsdir,datadirlabels{datadir}));
    end
  end
  if 0 % all IGCI methods in a single figure, for each datadir separately, eachpair [NOT IN ARTICLE, STILL USEFUL]
    methods = {'slope_','slope++_','entropy_KSG','entropy_Shannon_spacing_V', 'entropy_Shannon_spacing_Vb', 'entropy_Shannon_spacing_Vpconst', 'entropy_Shannon_spacing_Vplin', 'entropy_Shannon_spacing_Vplin2', 'entropy_Shannon_spacing_VKDE', 'entropy_Shannon_KDP','entropy_Shannon_PSD_SzegoT','entropy_Shannon_kNN_k','entropy_Shannon_Edgeworth','entropy_Shannon_MaxEnt1','entropy_Shannon_MaxEnt2','entropy_Shannon_expF'}; 
    % org_entropy: same as ent
    for thenorm=1:2
      fullmethods = {};
      for i=1:length(methods)
        fullmethods{i} = sprintf('igci_N=Inf_%s_norm%d',methods{i},thenorm);
      end
      if thenorm==1
        normlabel = 'uniform';
      else
        normlabel = 'Gaussian';
      end
      for datadir=1:length(datadirs)
        % eachpair
        plot_results_eachpair(datadirs{datadir},Inf,fullmethods,sprintf('%s_out/',datadirs{datadir}),1,translate(methods,'IGCI-'),strict);
        print('-depsc2',sprintf('%s/IGCI_eachpair_N=Inf_%s_%s.eps',plotsdir,normlabel,datadirlabels{datadir}));
      end
    end
  end

  % ANM
  ANMmethods = {'pHSIC_', 'HSIC_', 'HSIC__split', 'HSIC__fk', 'HSIC__split_fk', 'entropy_KSG', 'entropy_Shannon_spacing_V', 'entropy_Shannon_spacing_Vb', 'entropy_Shannon_spacing_Vpconst', 'entropy_Shannon_spacing_Vplin', 'entropy_Shannon_spacing_Vplin2', 'entropy_Shannon_KDP', 'entropy_Shannon_PSD_SzegoT', 'entropy_Shannon_spacing_VKDE', 'FN_', 'Gauss_', 'MML_', 'entropy_Shannon_kNN_k', 'entropy_Shannon_Edgeworth', 'entropy_Shannon_MaxEnt1', 'entropy_Shannon_MaxEnt2', 'entropy_Shannon_expF'};
  if 0 % details for each method
    methods = ANMmethods; 
    grouplabels = datadirlabels;
    variants = {'','_disc','_undisc','_dist=1e-9'};
    variantlabels = {'unperturbed','discretized','undiscretized','small noise'};
    for themethod=1:length(methods)
      thedatadirs = cell(length(grouplabels)*length(variantlabels),1);
      themethods = cell(length(grouplabels)*length(variantlabels),1);
      for group = 1:length(grouplabels)
        for variant = 1:length(variantlabels)
          method = (group-1)*length(variantlabels) + variant;
          thedatadirs{method} = datadirs{group};
          themethods{method} = [thedatadirs{method} '_out/' 'anm_N=Inf_FITC=100_' methods{themethod} '_lbfgsb' variants{variant}];
        end
      end

      for thetype=1:length(types)
        type = types{thetype};
        figure;
        subplot(2,2,1);
        plot_results_bars(thedatadirs,Inf,themethods,grouplabels,variantlabels,strict,0,type);
        title([translate(methods{themethod},'ANM-')]);
        print('-depsc2',[plotsdir '/ANM-' translate(methods{themethod},'med') '_N=Inf_FITC=100_lbfgsb_' type '.eps']);
      end
    end
  end

  % IGCI
  IGCImethods = {'org_entropy_','slope_','slope++_','entropy_KSG','entropy_Shannon_kNN_k','entropy_Shannon_Edgeworth','entropy_Shannon_spacing_V','entropy_Shannon_spacing_Vb','entropy_Shannon_spacing_Vpconst','entropy_Shannon_spacing_Vplin','entropy_Shannon_spacing_Vplin2','entropy_Shannon_KDP','entropy_Shannon_MaxEnt1','entropy_Shannon_MaxEnt2','entropy_Shannon_PSD_SzegoT','entropy_Shannon_expF','entropy_Shannon_spacing_VKDE'};
  if 0 % details for each method
    methods = IGCImethods;
    grouplabels = datadirlabels;
    variants = {'norm1','norm1_disc','norm1_undisc','norm1_dist=1e-9','norm2','norm2_disc','norm2_undisc','norm2_dist=1e-9'};
    variantlabels = {'uniform, unperturbed','uniform, discretized','uniform, undiscretized','uniform, small noise','Gaussian,unperturbed','Gaussian,discretized','Gaussian,undiscretized','Gaussian,small noise'};
    for themethod=1:length(methods)
      thedatadirs = cell(length(grouplabels)*length(variantlabels),1);
      themethods = cell(length(grouplabels)*length(variantlabels),1);
      for group = 1:length(grouplabels)
        for variant = 1:length(variantlabels)
          method = (group-1)*length(variantlabels) + variant;
          thedatadirs{method} = datadirs{group};
          themethods{method} = [thedatadirs{method} '_out/' 'igci_N=Inf_' methods{themethod} '_' variants{variant}];
        end
      end

      for thetype=1:length(types)
        type = types{thetype};
        figure;
        subplot(2,2,1);
        plot_results_bars(thedatadirs,Inf,themethods,grouplabels,variantlabels,strict,0,type);
        title([translate(methods{themethod},'IGCI-')]);
        print('-depsc2',[plotsdir '/IGCI_' translate(methods{themethod},med) '_N=Inf_' type '.eps']);
      end
    end
  end

  if 0 % anm, vary FITC
    % TODO:   make plot using eachpair instead
    % TODO:   combine more methods into plots?

    grouplabels = datadirlabels;
    variants = {'FITC=100','FITC=200','FITC=500','FITC=1000'};
    variantlabels = {'FITC=100','FITC=200','FITC=500','FITC=1000'};
    
    methods = {'pHSIC_','HSIC_','entropy_Shannon_KDP','entropy_Shannon_PSD_SzegoT','MML_'};
    methodlabels = translate(methods,'ANM-');
    titles = {'ANM-pHSIC','ANM-HSIC','ANM-ent-KDP','ANM-ent-PSD','ANM-MML'};
    for m=1:length(methods)
      plot_methods_variants(methods{m},grouplabels,variants,variantlabels,datadirs,'_out/anm_N=Inf_','_','_lbfgsb',strict,[plotsdir '/ANM_N=Inf_' methods{m} '_varyFITC'],titles{m},types);
    end
    % pHSIC-min is a special case... (uses minimize.m instead of lbfgsb)
    plot_methods_variants('pHSIC_',grouplabels,variants,variantlabels,datadirs,'_out/anm_N=Inf_','_','_minimize',strict,[plotsdir '/ANM_N=Inf_pHSICmin_varyFITC'],['ANM-pHSIC-min'],types);
  end

  if 0 % anm, vary Nmax
    % TODO:   make plot using eachpair instead
    % TODO:   combine more methods into plots?

    methods = {'pHSIC_','HSIC_','entropy_Shannon_KDP','entropy_Shannon_PSD_SzegoT','MML_'};
    methodlabels = translate(methods,'IGCI-');
    grouplabels = datadirlabels;
    variants = {'N=100','N=200','N=500','N=1000'};
    variantlabels = {'Nmax=100','Nmax=200','Nmax=500','Nmax=1000'};
    for m=1:length(methods)
      plot_methods_variants(methods{m},grouplabels,variants,variantlabels,datadirs,'_out/anm_','_FITC=0_','_lbfgsb',strict,[plotsdir '/ANM_FITC=0_' methods{m} '_varyN'],['ANM-' methodlabels{m} ', FITC=0'],types);
    end
    variants = {'N=100','N=200','N=500','N=1000','N=Inf'};
    variantlabels = {'Nmax=100','Nmax=200','Nmax=500','Nmax=1000','Nmax=Inf'};
    for m=1:length(methods)
      plot_methods_variants(methods{m},grouplabels,variants,variantlabels,datadirs,'_out/anm_','_FITC=100_','_lbfgsb',strict,[plotsdir '/ANM_FITC=100_' methods{m} '_varyN'],['ANM-' methodlabels{m} ', FITC=100'],types);
    end
  end

  if 0 % igci, vary Nmax
    % TODO:   make plot using eachpair instead
    % TODO:   combine more methods into plots?

    methods = {'org_entropy_', 'slope_', 'slope++_', 'entropy_KSG', 'entropy_Shannon_MaxEnt1', 'entropy_Shannon_MaxEnt2', 'entropy_Shannon_Edgeworth', 'entropy_Shannon_expF'};
    %Shannon_kNN_k Shannon_spacing_V Shannon_spacing_Vb Shannon_spacing_Vpconst Shannon_spacing_Vplin Shannon_spacing_Vplin2 Shannon_KDP Shannon_PSD_SzegoT Shannon_spacing_VKDE
    methodlabels = translate(methods,'IGCI-');
    grouplabels = datadirlabels;
    variants = {'N=10','N=50','N=100','N=200','N=500','N=1000','N=Inf'};
    variantlabels = {'Nmax=10','Nmax=50','Nmax=100','Nmax=200','Nmax=500','Nmax=1000','Nmax=Inf'};
    for m=1:length(methods)
      plot_methods_variants(methods{m},grouplabels,variants,variantlabels,datadirs,'_out/igci_','_','_norm1',strict,[plotsdir '/IGCI_' methods{m} '_norm1_varyN'],['IGCI-' methodlabels{m}],types);
      plot_methods_variants(methods{m},grouplabels,variants,variantlabels,datadirs,'_out/igci_','_','_norm2',strict,[plotsdir '/IGCI_' methods{m} '_norm2_varyN'],['IGCI-' methodlabels{m}],types);
    end
  end

  % curves
  if 0
    figure
    for i=1:length(datadirs)
      subplot(2,3,i);
      plot_results_sigcurve(datadirs{i},Inf,{'anm_N=Inf_FITC=100_pHSIC__lbfgsb'},[datadirs{i} '_out/'],30,'ANM-pHSIC',0,0);
      title([datadirlabels{i} ', ANM-pHSIC']);
    end
    print('-depsc2',[plotsdir '/ANM_N=Inf_pHSIC_curve.eps']);
  end

  % ANM ROC curves
  ANMmethods = {'pHSIC_', 'HSIC_', 'HSIC__split', 'HSIC__fk', 'HSIC__split_fk', 'entropy_KSG', 'entropy_Shannon_spacing_V', 'entropy_Shannon_spacing_Vb', 'entropy_Shannon_spacing_Vpconst', 'entropy_Shannon_spacing_Vplin', 'entropy_Shannon_spacing_Vplin2', 'entropy_Shannon_KDP', 'entropy_Shannon_PSD_SzegoT', 'entropy_Shannon_spacing_VKDE', 'FN_', 'Gauss_', 'MML_', 'entropy_Shannon_kNN_k', 'entropy_Shannon_Edgeworth', 'entropy_Shannon_MaxEnt1', 'entropy_Shannon_MaxEnt2', 'entropy_Shannon_expF'};
  if 1 % curves for each method
    methods = ANMmethods;
    linecolors = [0,0,1;0,0,1;0,0,1;0,0,1;0,0.5,0;1,0,0;0,0.75,0.75;0.75,0,0.75]; %0.75,0.75,0;0.25,0.25,0.25;0,1,0;0,0,0];
    linestyles = {'-','--',':','-.','-','-','-','-'};
    linewidths = {2.0,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
    for themethod = 1:length(methods);
      figure;
      subplot(2,2,1);
      thedatadirs = {datadirs{1}, datadirs{1}, datadirs{1}, datadirs{1}, datadirs{2}, datadirs{3}, datadirs{4}, datadirs{5}};
      themethods = cell(8,1);
      themethods{1} = [datadirs{1} '_out/' 'anm_N=Inf_FITC=100_' methods{themethod} '_lbfgsb'];
      themethods{2} = [datadirs{1} '_out/' 'anm_N=Inf_FITC=100_' methods{themethod} '_lbfgsb_disc'];
      themethods{3} = [datadirs{1} '_out/' 'anm_N=Inf_FITC=100_' methods{themethod} '_lbfgsb_undisc'];
      themethods{4} = [datadirs{1} '_out/' 'anm_N=Inf_FITC=100_' methods{themethod} '_lbfgsb_dist=1e-9'];
      themethods{5} = [datadirs{2} '_out/' 'anm_N=Inf_FITC=100_' methods{themethod} '_lbfgsb'];
      themethods{6} = [datadirs{3} '_out/' 'anm_N=Inf_FITC=100_' methods{themethod} '_lbfgsb'];
      themethods{7} = [datadirs{4} '_out/' 'anm_N=Inf_FITC=100_' methods{themethod} '_lbfgsb'];
      themethods{8} = [datadirs{5} '_out/' 'anm_N=Inf_FITC=100_' methods{themethod} '_lbfgsb'];
      thelabels = cell(8,1);
      thelabels{1} = [datadirlabels{1}];
      thelabels{2} = [datadirlabels{1} ' (discretized)'];
      thelabels{3} = [datadirlabels{1} ' (undiscretized)'];
      thelabels{4} = [datadirlabels{1} ' (small noise)'];
      thelabels{5} = datadirlabels{2};
      thelabels{6} = datadirlabels{3};
      thelabels{7} = datadirlabels{4};
      thelabels{8} = datadirlabels{5};
      plot_results_roccurves(thedatadirs,Inf,themethods,thelabels,strict,[0.5,0.8],1,linecolors,linestyles,linewidths);
      title([translate(methods{themethod},'ANM-')]);
      print('-depsc2',[plotsdir '/ANM-' translate(methods{themethod},'med') '_N=Inf_FITC=100_lbfgsb_roccurves.eps']);
    end
  end

  % IGCI ROC curves
  IGCImethods = {'org_entropy_','slope_','slope++_','entropy_KSG','entropy_Shannon_kNN_k','entropy_Shannon_Edgeworth','entropy_Shannon_spacing_V','entropy_Shannon_spacing_Vb','entropy_Shannon_spacing_Vpconst','entropy_Shannon_spacing_Vplin','entropy_Shannon_spacing_Vplin2','entropy_Shannon_KDP','entropy_Shannon_MaxEnt1','entropy_Shannon_MaxEnt2','entropy_Shannon_PSD_SzegoT','entropy_Shannon_expF','entropy_Shannon_spacing_VKDE'};
  if 1 % curves for each method
    methods = IGCImethods;
    linecolors = [0,0,1;0,0,1;0,0,1;0,0,1;0,0.5,0;1,0,0;0,0.75,0.75;0.75,0,0.75]; %0.75,0.75,0;0.25,0.25,0.25;0,1,0;0,0,0];
    linestyles = {'-','--',':','-.','-','-','-','-'};
    linewidths = {2.0,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
    for thenorm=1:2
      normname = sprintf('_norm%d', thenorm);
      if thenorm == 1
        normlabel = 'uniform';
      else
        normlabel = 'Gaussian';
      end
      for themethod = 1:length(methods);
        figure;
        subplot(2,2,1);
        thedatadirs = {datadirs{1}, datadirs{1}, datadirs{1}, datadirs{1}, datadirs{2}, datadirs{3}, datadirs{4}, datadirs{5}};
        themethods = cell(8,1);
        themethods{1} = [datadirs{1} '_out/' 'igci_N=Inf_' methods{themethod} normname];
        themethods{2} = [datadirs{1} '_out/' 'igci_N=Inf_' methods{themethod} normname '_disc'];
        themethods{3} = [datadirs{1} '_out/' 'igci_N=Inf_' methods{themethod} normname '_undisc'];
        themethods{4} = [datadirs{1} '_out/' 'igci_N=Inf_' methods{themethod} normname '_dist=1e-9'];
        themethods{5} = [datadirs{2} '_out/' 'igci_N=Inf_' methods{themethod} normname];
        themethods{6} = [datadirs{3} '_out/' 'igci_N=Inf_' methods{themethod} normname];
        themethods{7} = [datadirs{4} '_out/' 'igci_N=Inf_' methods{themethod} normname];
        themethods{8} = [datadirs{5} '_out/' 'igci_N=Inf_' methods{themethod} normname];
        thelabels = cell(8,1);
        thelabels{1} = [datadirlabels{1}];
        thelabels{2} = [datadirlabels{1} ' (discretized)'];
        thelabels{3} = [datadirlabels{1} ' (undiscretized)'];
        thelabels{4} = [datadirlabels{1} ' (small noise)'];
        thelabels{5} = datadirlabels{2};
        thelabels{6} = datadirlabels{3};
        thelabels{7} = datadirlabels{4};
        thelabels{8} = datadirlabels{5};
        plot_results_roccurves(thedatadirs,Inf,themethods,thelabels,strict,[0.5,0.8],1,linecolors,linestyles,linewidths);
        title([translate(methods{themethod},'IGCI-') ' ' normlabel]);
        print('-depsc2',[plotsdir '/IGCI-' translate(methods{themethod},'med') '_N=Inf_' normlabel '_roccurves.eps']);
      end
    end
  end

  % ANM ROC curves
  ANMmethods = {'pHSIC_', 'HSIC_', 'HSIC__split', 'HSIC__fk', 'HSIC__split_fk', 'entropy_KSG', 'entropy_Shannon_spacing_V', 'entropy_Shannon_spacing_Vb', 'entropy_Shannon_spacing_Vpconst', 'entropy_Shannon_spacing_Vplin', 'entropy_Shannon_spacing_Vplin2', 'entropy_Shannon_KDP', 'entropy_Shannon_PSD_SzegoT', 'entropy_Shannon_spacing_VKDE', 'FN_', 'Gauss_', 'MML_', 'entropy_Shannon_kNN_k', 'entropy_Shannon_Edgeworth', 'entropy_Shannon_MaxEnt1', 'entropy_Shannon_MaxEnt2', 'entropy_Shannon_expF'};
  if 1 % all curves
    methods = ANMmethods;
    thedatadirs = cell(length(methods),1);
    themethods = cell(length(methods),1);
    thelabels = cell(length(methods),1);
    linecolors = zeros(length(methods),3);
    linestyles = cell(length(methods),1);
    linewidths = cell(length(methods),1);
    for themethod = 1:length(methods)
      thedatadirs{themethod} = datadirs{1};
      linecolors(themethod,:) = [0,0,1];
      linestyles{themethod} = '-';
      linewidths{themethod} = 0.5;
      themethods{themethod} = [thedatadirs{themethod} '_out/' 'anm_N=Inf_FITC=100_' methods{themethod} '_lbfgsb'];
      thelabels{themethod} = [translate(methods{themethod},'ANM-')];
    end
    figure;
    subplot(2,2,1);
    plot_results_roccurves(thedatadirs,Inf,themethods,thelabels,strict,0,1,linecolors,linestyles,linewidths);
    title('ANM');
    print('-depsc2',[plotsdir '/ANM_all_N=Inf_FITC=100_lbfgsb_roccurves.eps']);
  end

  % IGCI ROC curves
  IGCImethods = {'org_entropy_','slope_','slope++_','entropy_KSG','entropy_Shannon_kNN_k','entropy_Shannon_Edgeworth','entropy_Shannon_spacing_V','entropy_Shannon_spacing_Vb','entropy_Shannon_spacing_Vpconst','entropy_Shannon_spacing_Vplin','entropy_Shannon_spacing_Vplin2','entropy_Shannon_KDP','entropy_Shannon_MaxEnt1','entropy_Shannon_MaxEnt2','entropy_Shannon_PSD_SzegoT','entropy_Shannon_expF','entropy_Shannon_spacing_VKDE'};
  if 1 % all curves
    methods = IGCImethods;
    for thenorm=1:2
      normname = sprintf('_norm%d', thenorm);
      if thenorm == 1
        normlabel = 'uniform';
      else
        normlabel = 'Gaussian';
      end
      thedatadirs = cell(length(methods),1);
      themethods = cell(length(methods),1);
      thelabels = cell(length(methods),1);
      linecolors = zeros(length(methods),3);
      linestyles = cell(length(methods),1);
      linewidths = cell(length(methods),1);
      for themethod = 1:length(methods)
        thedatadirs{themethod} = datadirs{1};
        linecolors(themethod,:) = [0,0,1];
        linestyles{themethod} = '-';
        linewidths{themethod} = 0.5;
        themethods{themethod} = [thedatadirs{themethod} '_out/' 'igci_N=Inf_' methods{themethod} normname];
        thelabels{themethod} = [translate(methods{themethod},'IGCI-')];
      end
      figure;
      subplot(2,2,1);
      plot_results_roccurves(thedatadirs,Inf,themethods,thelabels,strict,0,1,linecolors,linestyles,linewidths);
      title(['IGCI ' normlabel]);
      print('-depsc2',[plotsdir '/IGCI_all_N=Inf_' normlabel '_roccurves.eps']);
    end
  end

  % counts
  if 0
    grouplabels = datadirlabels;
    variants = {'','_disc','_undisc','_dist=1e-9'};
    variantlabels = {'unperturbed','discretized','undiscretized','small noise'};
    thedatadirs = cell(length(grouplabels)*length(variantlabels),1);
    themethods = cell(length(grouplabels)*length(variantlabels),1);
    for group = 1:length(grouplabels)
      for variant = 1:length(variantlabels)
        method = (group-1)*length(variantlabels) + variant;
        thedatadirs{method} = datadirs{group};
        themethods{method} = [thedatadirs{method} '_out/' 'count_N=Inf' variants{variant}];
      end
    end
    for thetype=1:length(types)
      type = types{thetype};
      figure;
      subplot(2,2,1);
      plot_results_bars(thedatadirs,Inf,themethods,grouplabels,variantlabels,strict,0,type);
      title('count');
      print('-depsc2',[plotsdir '/count_' type '.eps']);
    end
  end

return
    

function plot_methods(methods,grouplabels,datadirs,variantlabels,prefix,suffix,strict,epsfilebase,tit,types);
  thedatadirs = cell(length(grouplabels)*length(variantlabels),1);
  themethods = cell(length(grouplabels)*length(variantlabels),1);
  for group = 1:length(grouplabels)
    for variant = 1:length(variantlabels)
      index = (group-1)*length(variantlabels) + variant;
      thedatadirs{index} = datadirs{variant};
      themethods{index} = [thedatadirs{index} prefix methods{group} suffix];
    end
  end
  for thetype=1:length(types)
    type = types{thetype};
    figure;
    subplot(2,2,1);
    plot_results_bars(thedatadirs,Inf,themethods,grouplabels,variantlabels,strict,0,type);
    title(tit);
    print('-depsc2',[epsfilebase '_' type '.eps']);
  end
return
    
function plot_methods_perturbations(methods,grouplabels,variants,variantlabels,datadir,prefix,midfix,suffix,strict,epsfilebase,tit,types);
  thedatadirs = cell(length(grouplabels)*length(variantlabels),1);
  themethods = cell(length(grouplabels)*length(variantlabels),1);
  for group = 1:length(grouplabels)
    for variant = 1:length(variantlabels)
      index = (group-1)*length(variantlabels) + variant;
      thedatadirs{index} = datadir;
      themethods{index} = [thedatadirs{index} prefix methods{group} midfix variants{variant} suffix];
    end
  end
  for thetype=1:length(types)
    type = types{thetype};
    figure;
    subplot(2,2,1);
    plot_results_bars(thedatadirs,Inf,themethods,grouplabels,variantlabels,strict,0,type);
    title(tit);
    print('-depsc2',[epsfilebase '_' type '.eps']);
  end
return

function plot_methods_variants(method,grouplabels,variants,variantlabels,datadirs,prefix,midfix,suffix,strict,epsfilebase,tit,types);
  thedatadirs = cell(length(grouplabels)*length(variantlabels),1);
  themethods = cell(length(grouplabels)*length(variantlabels),1);
  for group = 1:length(grouplabels)
    for variant = 1:length(variantlabels)
      index = (group-1)*length(variantlabels) + variant;
      thedatadirs{index} = datadirs{group};
      themethods{index} = [thedatadirs{index} prefix variants{variant} midfix method suffix];
    end
  end
  for thetype=1:length(types)
    type = types{thetype};
    figure;
    subplot(2,2,1);
    plot_results_bars(thedatadirs,Inf,themethods,grouplabels,variantlabels,strict,0,type);
    title(tit);
    print('-depsc2',[epsfilebase '_' type '.eps']);
  end
return

function labels = translate(methodnames,verbosity)
  if ischar(methodnames)
    mn{1} = methodnames;
  else
    mn = methodnames;
  end

  if( nargin == 1 )
    verbosity = '';
  end

  labels = cell(length(mn),1);
  themethods = {'pHSIC_', 'HSIC_', 'HSIC__split', 'HSIC__fk', 'HSIC__split_fk', 'entropy_KSG', 'entropy_Shannon_spacing_V', 'entropy_Shannon_spacing_Vb', 'entropy_Shannon_spacing_Vpconst', 'entropy_Shannon_spacing_Vplin', 'entropy_Shannon_spacing_Vplin2', 'entropy_Shannon_spacing_VKDE', 'entropy_Shannon_KDP', 'entropy_Shannon_PSD_SzegoT', 'entropy_Shannon_kNN_k', 'entropy_Shannon_Edgeworth', 'entropy_Shannon_MaxEnt1', 'entropy_Shannon_MaxEnt2', 'entropy_Shannon_expF', 'FN_', 'Gauss_', 'MML_', 'org_entropy_', 'slope_', 'slope++_'};
  if( strcmp(verbosity,'') )
    thelabels = {'pHSIC', 'HSIC', 'HSIC-ds', 'HSIC-fk', 'HSIC-ds-fk', '1sp', 'sp1', 'sp2', 'sp3', 'sp4', 'sp5', 'sp6', 'KDP', 'PSD', '3NN', 'EdE', 'ME1', 'ME2', 'Gau', 'FN', 'Gauss', 'MML', 'org-ent', 'slope', 'slope++'};
  elseif( strcmp(verbosity,'med') )
    thelabels = {'pHSIC', 'HSIC', 'HSIC-ds', 'HSIC-fk', 'HSIC-ds-fk', 'ent-1sp', 'ent-sp1', 'ent-sp2', 'ent-sp3', 'ent-sp4', 'ent-sp5', 'ent-sp6', 'ent-KDP', 'ent-PSD', 'ent-3NN', 'ent-EdE', 'ent-ME1', 'ent-ME2', 'ent-Gau', 'FN', 'Gauss', 'MML', 'org-ent', 'slope', 'slope++'};
  else
    thelabels = {'ANM-pHSIC', 'ANM-HSIC', 'ANM-HSIC-ds', 'ANM-HSIC-fk', 'ANM-HSIC-ds-fk', [verbosity 'ent-1sp'], [verbosity 'ent-sp1'], [verbosity 'ent-sp2'], [verbosity 'ent-sp3'], [verbosity 'ent-sp4'], [verbosity 'ent-sp5'], [verbosity 'ent-sp6'], [verbosity 'ent-KDP'], [verbosity 'ent-PSD'], [verbosity 'ent-3NN'], [verbosity 'ent-EdE'], [verbosity 'ent-ME1'], [verbosity 'ent-ME2'], [verbosity 'ent-Gau'], 'ANM-FN', 'ANM-Gauss', 'ANM-MML', 'IGCI-org-ent', 'IGCI-slope', 'IGCI-slope++'};
  end
  for i=1:length(mn)
    labels{i} = 'unknown';
    for j=1:length(themethods)
      if strcmp(mn{i},themethods{j})
        labels{i} = thelabels{j};
      end
    end
  end

  if ischar(methodnames)
    labels = labels{1};
  end
return
