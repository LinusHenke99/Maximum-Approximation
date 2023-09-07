# Error approximation

This repo is dedicated to estimating the maximum value of a batch with N components using Chebyshev approximation.

## Project Structure
The source directory of this project contains multiple python scripts, some of them for generating datasets, some of them for plotting. The `data` folder contains multiple
datasets and their approximation of the maximum values of a given batch using chebyshev approximation.

### `gauss_error_accumulation.py`
This python script can be used to generate data using a gaussian distribution, which then is saved as a json file in the `data` folder. The values are hardcoded in the script.

## Conclusions
Here is a short presentation of the Conclusions I have drawn so far.

### Optimal values for the approximation borders of the Chebyshev approximation for gaussian distributed inputs
The task here was to find the optimal factor $\mu$

