function simulate_pairs(N,datadir,numpairs,confprob,output,seed,AB_wEX,AB_wEY,AB_wEZ,AB_sEX,AB_sEY,AB_sEZ,AB_sX,AB_epsX,AB_epsY,jitter)
% function simulate_pairs(N,datadir,numpairs,confprob,output,seed,AB_wEX,AB_wEY,AB_wEZ,AB_sEX,AB_sEY,AB_sEZ,AB_sX,AB_epsX,AB_epsY,jitter)
%
% Simulate cause-effect pairs and save them or display them as scatter plots
%
% INPUT:
%   N                     number of data points per pair (e.g., 1000)
%   datadir               directory where the cause-effect-pair data will be written
%                           this directory will contain a file 'pairmeta.txt' with the metadata
%                           and then files pair0001.txt, pair0002.txt, ... with the actual data
%                           as ASCII text files (readable with load('-ascii',...))
%   numpairs              number of pairs to generate (e.g., 100)
%   confprob              vector with probability of number of confounders;
%                           confprob(i) is the probability of i-1 confounders
%                           for example, confprob == [1] means zero confounders with probability 1
%   output                possibilities: {'files','scatterplots'}
%   seed                  random number seed
%   AB_wEX                Gamma parameters for $E_X$; in article: $(a_{w_{E_X}}, b_{w_{E_X}})$; example: [5,0.1]
%   AB_wEY                Gamma parameters for $E_Y$; in article: $(a_{w_{E_Y}}, b_{w_{E_Y}})$; example: [5,0.1]
%   AB_wEZ                Gamma parameters for $E_Z$; in article: $(a_{w_{E_Z}}, b_{w_{E_Z}})$; example: [5,0.1]
%   AB_sEX                Gamma parameters for $\sigma_{E_X}$; in article: $(a_{\sigma_{E_X}},b_{\sigma_{E_X}})$; example: [2,1.5]
%   AB_sEY                Gamma parameters for $\sigma_{E_Y}$; in article: $(a_{\sigma_{E_Y}},b_{\sigma_{E_Y}})$; example: [2,15]
%   AB_sEZ                Gamma parameters for $\sigma_{E_Z}$; in article: $(a_{\sigma_{E_Z}},b_{\sigma_{E_Z}})$; example: [2,15]
%   AB_sX                 Gamma parameters for $\sigma_X$; in article: $(a_{\sigma_{X}},b_{\sigma_{X}})$; example: [2,15]
%   AB_epsX               Gamma parameters for $\sigma_{\epsilon_X}$; in article: $(a_{\sigma_{\epsilon_X}},b_{\sigma_{\epsilon_X}})$; example: [2,0.1]
%   AB_epsY               Gamma parameters for $\sigma_{\epsilon_Y}$; in article: $(a_{\sigma_{\epsilon_Y}},b_{\sigma_{\epsilon_Y}})$; example: [2,0.1]
%   jitter                jitter for random GP; in article: $\tau$; example: 1e-4
%
% The sampling process and the parameters are explained in the article.
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

% Test output argument
if ~strcmp(output,'files') && ~strcmp(output,'scatterplots')
  error('output argument should be ''files'' or ''scatterplots''.');
end

% init (set paths)
curpath = path;
global CEP_PATH_UTIL;
addpath(CEP_PATH_UTIL); 
global CEP_PATH_GPML;
run([CEP_PATH_GPML '/startup']); 

% Create output directory
[success,msg] = mkdir(datadir);

% Set random number seed
s = RandStream('mcg16807','Seed',seed);
RandStream.setDefaultStream(s);
rand(1,1)

% Simulate pairs
dirs = zeros(numpairs,1);
for pair=1:numpairs

  fprintf('Simulating pair %d of %d...',pair,numpairs);

  % Draw noise variables EX and EY
  wEX = gamrnd(AB_wEX(1),AB_wEX(2));
%  wEX = .1 + rand(1,1) * 0.9;
  EX = randmonoinc(randn(N,1),wEX,1,jitter);
  EX = (EX - mean(EX)) / std(EX);
  wEY = gamrnd(AB_wEY(1),AB_wEY(2));
%  wEY = .1 + rand(1,1) * 0.9;
  EY = randmonoinc(randn(N,1),wEY,1,jitter);
  EY = (EY - mean(EY)) / std(EY);

  % Draw number of confounders randomly
  numZ = draw(1,length(confprob),confprob) - 1;
  fprintf('%d confounders\n',numZ);

  % Draw confounder variables EZ
  EZ = zeros(N,numZ);
  for i=1:numZ
    wEZ = gamrnd(AB_wEZ(1),AB_wEZ(2));
%    wEZ = .1 + rand(1,1) * 0.9;
    EZ(:,i) = randmonoinc(randn(N,1),wEZ,1,1e-3);
    EZ(:,i) = (EZ(:,i) - mean(EZ(:,i))) / std(EZ(:,i));
  end

  % Draw function
  %X = EX;
% sigma_EX = exp(randn(1,1)) + 1;
  sigma_EX = gamrnd(AB_sEX(1),AB_sEX(2));
% sigma_EZ = exp(randn(1,numZ)) + 1) * sigma_confounder;
  sigma_EZ = gamrnd(ones(1,numZ) * AB_sEZ(1),ones(1,numZ) * AB_sEZ(2));
  X = randgp([EX,EZ],[sigma_EX,sigma_EZ],1,jitter);
  X = (X - mean(X)) / std(X);
  [h,p] = kstest(X)
%  Y = randgp([X,EY,EZ],[2+exp(randn(1,1)),1+exp(randn(1,1)),exp(randn(1,numZ))+1],1,jitter);
%  Y = randgp([X,EY,EZ],[5*(1+exp(randn(1,1))),(1/noisestrength)*(1+exp(randn(1,1))),exp(randn(1,numZ))+1],1,jitter);
%  sigma_X = (exp(randn(1,1)) + 1) * sigma_cause;
  sigma_X = gamrnd(AB_sX(1),AB_sX(2));
%  sigma_EY = (exp(randn(1,1)) + 1) * sigma_noise;
  sigma_EY = gamrnd(AB_sEY(1),AB_sEY(2));
%  sigma_EZ = (exp(randn(1,numZ)) + 1) * sigma_confounder;
  sigma_EZ = gamrnd(ones(1,numZ) * AB_sEZ(1),ones(1,numZ) * AB_sEZ(2));
  Y = randgp([X,EY,EZ],[sigma_X,sigma_EY,sigma_EZ],1,jitter);
  Y = (Y - mean(Y)) / std(Y);

  % Add Gaussian measurment noise variables mX, mY
%  mX = randn(N,1) * measurementnoise * (exp(randn(1,1))+1);
%  mY = randn(N,1) * measurementnoise * (exp(randn(1,1))+1);
  sigma_epsX = gamrnd(AB_epsX(1),AB_epsX(2));
  sigma_epsY = gamrnd(AB_epsY(1),AB_epsY(2));
  epsX = randn(N,1) * sigma_epsX;
  epsY = randn(N,1) * sigma_epsY;
  X = X + epsX;
  Y = Y + epsY;

  % Some more stuff to add in the future:
  %   nonGaussian measurement noise
  %   measurement distortion

  % Flip a coin and swap X and Y if heads comes up
  if rand(1,1) < 0.5
    data = [X,Y];
    dirs(pair) = 1;
  else
    data = [Y,X];
    dirs(pair) = 2;
  end

  % Write file
  if strcmp(output,'files')
    filename = sprintf('%s/pair%04d.txt',datadir,pair);
    save(filename,'-ascii','data');
  end

  % Display scatter plots
  if strcmp(output,'scatterplots')
    scatter(X,Y);
    if dirs(pair) == 1
      xlabel('cause');
      ylabel('effect');
    else
      xlabel('effect');
      ylabel('cause');
    end
    pause
  end
end

% Write metadata
if strcmp(output,'files')
  filename = sprintf('%s/pairmeta.txt',datadir);
  fid = fopen(filename,'w');
  for pair=1:numpairs
    if dirs(pair) == 1
      fprintf(fid,'%04d %d %d %d %d %d\n',pair,1,1,2,2,1);
    else
      fprintf(fid,'%04d %d %d %d %d %d\n',pair,2,2,1,1,1);
    end
  end
  fclose(fid);
end

% restore path
path(curpath);

return
