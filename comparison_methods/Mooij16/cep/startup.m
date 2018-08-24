% Typically, the user should not change this startup.m file, but rather local_config.m
% to specify the local directories into which external libraries have been installed
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.


% Use trick from Hannes Nickisch (GPML software) to autodetect local directory
OCT = exist('OCTAVE_VERSION') ~= 0;           % check if we run Matlab or Octave

me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
if OCT && numel(mydir)==2 
  if strcmp(mydir,'./'), mydir = [pwd,mydir(2:end)]; end
end                 % OCTAVE 3.0.x relative, MATLAB and newer have absolute path
global CEP_PATH;
CEP_PATH = mydir;
clear me mydir

% this directory is shared by all
addpath([CEP_PATH,'.'])

local_config;
