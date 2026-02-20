# Overview

modular/parallelized rewrite of previous notebook implementation of the composite model. 

## optimization

just as with the floquet matrix elements code, we run partitioned sweeps with process level parallelism. I am also going to try adding time evaluations on the following:
- how much wall time each partition takes
- timing for buiding the fluxonium, running floquet, and saving
- also timing within the model build, static portion (i.e. diagonalizing fluxonium, the chi to amp conversion, and constructing/transforming the operators) 


Note: make sure number of threads is set to 1 per worker. also make sure ft.Options(num_cpus = 1)

if using the floquet internal parallism, then max_worker = 1 and num_cpus = whatever value. 

## usage

root of project folder: pip install -e .

then we can run the scripts in the scripts folder:
1. for drive sweep (omega_d vs chi_ac at fixed flux)
python scripts/run_drive_sweep_parallel.py --outdir {date}_drive_sweep
2. flux sweep 
python scripts/run_flux_sweep_parallel.py --outdir {date}_flux_sweep
3. truncation metrics
python scripts/run_truncation_metrics_parallel.py --outdir {date}_truncation


after running script, it should output a file merging all the computed chunks!