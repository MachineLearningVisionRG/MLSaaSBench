# MLSaaSBench

## Instructions
The preprocessing is in two scripts, one for each dataset. The resulting results are the "preprocessed data".

We then use these as input to the training-evaluation which runs on different machines.

As it is, it reads the preprocessed data from the google cloud where we have uploaded them after we ran the preprocessed.

On the pc it reads them locally. (It is in the script commented).

For codecarbon it has a parameter for the region that we change according to the region of the google cloud vm.

The results are saved in csv files, one total for cross validation and one for the holdout (after the end of each training we save it with whatever results it has run so that it is lost).