##### Local configuration #####
 
# Specify C++ compiler
CC=g++34

# Specify MatLab MEX compiler
MEX=mex
MEXEXT=.mexa64

# Alternatively, for Octave:
# MEX=mkoctfile --mex
# MEXEXT=.mex

##### Makefile rules #####

all: fasthsic$(MEXEXT) 

# NOTE: the GNU scientific library needs to be installed

fasthsic$(MEXEXT): fasthsic.cpp hsic.cpp hsic.h
	$(MEX) -o $@ $< hsic.cpp -Ifasthsic -lgsl -lgslcblas

.PHONY : clean
clean:
	-rm fasthsic$(MEXEXT)
