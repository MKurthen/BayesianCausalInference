#!/bin/bash
#
# Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
# All rights reserved.  See the file LICENSE for license terms.

# Simulate data
N=1000
nrPairs=100

SIMDIR='SIM'
matlab -nodisplay -nodesktop -r "simulate_pairs($N,'$SIMDIR',$nrPairs,[1],'files',123,[5,0.1],[5,0.1],[5,0.1],[2,1.5],[2,1.5*10],[2,1.5*10],[2,1.5*2],[2,0.1],[2,0.1],1e-4); exit"
SIMDIR='SIM-c'
matlab -nodisplay -nodesktop -r "simulate_pairs($N,'$SIMDIR',$nrPairs,[0 1],'files',123,[5,0.1],[5,0.1],[5,0.1],[2,1.5],[2,1.5*10],[2,1.5*10],[2,1.5*2],[2,0.1],[2,0.1],1e-4); exit"
SIMDIR='SIM-ln'
matlab -nodisplay -nodesktop -r "simulate_pairs($N,'$SIMDIR',$nrPairs,[1],'files',123,[5,0.1],[5,0.1],[5,0.1],[2,1.5],[2,1.5*200],[2,1.5*10],[2,1.5*2],[2,0.01],[2,0.01],1e-4); exit"
SIMDIR='SIM-G'
matlab -nodisplay -nodesktop -r "simulate_pairs($N,'$SIMDIR',$nrPairs,[1],'files',123,[1e6,1e-3],[5,0.1],[5,0.1],[1e6,1e-3],[2,1.5*10],[2,1.5*10],[2,1.5*2],[2,0.1],[2,0.1],1e-4); exit"

#SIMDIR='simeleni'
#matlab -nodisplay -nodesktop -r "simulate_pairs_old($N,'$SIMDIR',$nrPairs,[1],2,10,10,0.1,'files',12345,1); exit"
