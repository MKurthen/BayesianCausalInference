function [working_environment_is_Matlab] = working_environment_Matlab()
%function [working_environment_is_Matlab] = working_environment_Matlab()
%Returns the current working environment. Sets the global variable
%g_working_environment_Matlab, if it has not been done before.
%
%OUTPUT:
%   working_environment_is_Matlab: 1 means 'the working enviroment is Matlab', 0 stands for 'the
%   working environment is Octave'. 

%Copyright (C) 2012- Zoltan Szabo ("http://www.gatsby.ucl.ac.uk/~szabo/", "zoltan (dot) szabo (at) gatsby (dot) ucl (dot) ac (dot) uk")
%
%This file is part of the ITE (Information Theoretical Estimators) toolbox.
%
%ITE is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
%the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
%
%This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License along with ITE. If not, see <http://www.gnu.org/licenses/>.

global g_working_environment_Matlab;
if isempty(g_working_environment_Matlab)%not yet set
    g_working_environment_Matlab = 1-size(ver('Octave'),1);%size(ver('Octave'),1) =0:Matlab, =1:Octave
end

working_environment_is_Matlab = g_working_environment_Matlab;
