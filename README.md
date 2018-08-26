# Prerequisits
    -Python 3.6.6
    -Colorama 0.3.9
    -Numpy 1.15.0
    -Scikit-Learn 0.19.1
    -SciPy 1.1.0

to use the NIFTy implementation:

    -NIFTy 5

the Jupyter notebooks havee been created using

    -jupyter 1.0.0

the Plots have been done using

    -matplotlib 2.2.3
    -seaborn 0.9.0

to use the methods for comparison:
  CGNN requires
  
    -tensorflow 1.10.0
    -joblib 0.12.2

  ANM-HSIC, ANM-MML, IGCI require

    -installed Matlab5 libraries, to use the Matlab API
      (see https://www.mathworks.com/help/matlab/matlab_external/get-started-with-matlab-engine-for-python.html )

# Benchmarks

to perform a benchmark, call the script do_benchmark.py via:

    python do_benchmark.py --args
  
where args refer to:

  --benchmark: one of the supported benchmarks, either a "bcs" 
        (bayesian causal sampling) benchmark, these are stored
        in the './benchmarks' folder and use the formatting as
        in the comparative study by Mooij16

  --model: currently either 1 or 2, 1 refers to the shallow model
        implemented via NumPy, 2 uses the same model implemented
        in NIFTy

  --nvar: value for the noise variance, 1e-2 has shown to provide
        good results

  --nbins: number of bins for the Poissonian discretization and 
        for the field approximation. 512 performs well here

  --power_spectrum_beta: the P_beta power spectrum. This has to be
        given in the format such that the string 
        ("lambda q: " + power_spectrum_beta) can be parsed to a
        valid lambda expression, e.g. "2048/(q**4 + 1)"

  --power_spectrum_f: the P_f power spectrum. Same formmating as
        above

  --rho: Value for rho in the inference model.

  --scale_max: all x and y data will be scaled to the interval 
      [0, scale_max]
    
