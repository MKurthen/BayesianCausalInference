% Local configuration
% Change these directories to reflect your local installation

global CEP_PATH;
% This one is detected automatically by calling startup.m
% so it doesn't need to be specified by the user

% The directories below reflect locations of external libraries
% and should be modified by the user in order to be able to use
% the cep_*.m files that make use of these libraries

global CEP_PATH_GPML;
CEP_PATH_GPML = [CEP_PATH '/../extern/gpml-matlab-v3.5-2014-12-08'];

global CEP_PATH_ITE;
CEP_PATH_ITE = [CEP_PATH '/../extern/ITE-0.58/code/'];

%global CEP_PATH_LINGAM;
%CEP_PATH_LINGAM = [CEP_PATH '/../../../../code-extern/lingam-1.4.2/code'];

global CEP_PATH_MMLMIXTURES;
CEP_PATH_MMLMIXTURES = [CEP_PATH '/../extern/mixturecode'];

global CEP_PATH_FASTHSIC;
CEP_PATH_FASTHSIC = [CEP_PATH '/../fasthsic'];

global CEP_PATH_GPI;
CEP_PATH_GPI = [CEP_PATH '/../gpi'];

%global CEP_PATH_HSICREG;
%CEP_PATH_HSICREG = [CEP_PATH '/../hsicreg'];

global CEP_PATH_IGCI;
CEP_PATH_IGCI = [CEP_PATH '/../igci'];

%global CEP_PATH_PNL;
%CEP_PATH_PNL = [CEP_PATH '/../pnl'];

%global CEP_PATH_HKRR;
%CEP_PATH_HKRR = [CEP_PATH '/../heteroschedastic'];

global CEP_PATH_UTIL;
CEP_PATH_UTIL = [CEP_PATH '/../util'];

global CEP_PATH_FIT;
CEP_PATH_FIT = [CEP_PATH '/../fit'];
