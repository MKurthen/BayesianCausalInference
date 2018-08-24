function plot_all_pairs(datadir,filename,pairs,ext)
% function plot_all_pairs(datadir,filename,pairs,ext)
% 
% Makes scatter plots of all CEP pairs in a directory
%
% INPUT:
%   datadir     directory where to look for CEP pairs
%   filename    name of the .pdf / .png file to be written
%   pairs       vector of pair ids to include
%   ext         determines output file type ('pdf' or 'png')
%
% OUTPUT:       written to filename in format ext
%
% Copyright (c) 2008-2015  Stefan Harmeling, Joris Mooij <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

% init (set paths)
curpath = path;
global CEP_PATH_UTIL;
addpath(CEP_PATH_UTIL); 

nrpairs = length(pairs);
nrcols = ceil(sqrt(nrpairs / sqrt(2)));
nrrows = ceil(nrpairs / nrcols);
sw = 1 ./ [nrrows, nrcols];
clf
for col = 1:nrcols
  for row = 1:nrrows
    pair = (col-1)*nrrows + row;
    if pair <= nrpairs
      % load data and apply preprocessing
      [X,Y] = load_pair(pairs(pair),[],datadir);
      X = normalize(X(:,1),1); X = X - min(X);
      Y = normalize(Y(:,1),1); Y = Y - min(Y);

      pos = [row-1, nrcols-col] .* sw;
      axes('position', [pos, sw]);
      plot(X,Y,'k.','Markersize', 2);
      xlim([-0.2,1.2]);
      ylim([-0.2,1.2]);
      t=text(1.0, 0.0, num2str(pairs(pair)), 'Units', 'normalized');
      set(t, 'HorizontalAlignment', 'right');
      set(t, 'VerticalAlignment', 'bottom');
      set(t, 'FontUnits', 'normalized');
      set(t, 'FontSize', 1/14);
      axis off;
    end
  end
end

if strcmp(ext,'pdf')
  set(gcf,'PaperType','A4');
  set(gcf,'PaperOrientation','landscape');
  set(gcf,'PaperUnits','normalized');
  set(gcf,'PaperPosition',[0.05 0.05 0.9 0.9]);
  print('-dpdf',filename);
elseif strcmp(ext,'png')
  set(gcf,'PaperType','A4');
  set(gcf,'PaperOrientation','landscape');
  set(gcf,'PaperUnits','normalized');
  set(gcf,'PaperPosition',[0.05 0.05 0.9 0.9]);
  print('-dpng','-r1200',filename);
else
  error('Unknown graphics extension');
end

% restore path
path(curpath);

return
