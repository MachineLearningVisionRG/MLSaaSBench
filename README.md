# MLSaaSBench

Machine learning (ML) methods have been successfully employed to support
decision-making for Software as a Service (SaaS) providers. While most of the
published   research   primarily   emphasizes   prediction   accuracy,   other
important aspects, such as cloud deployment efficiency and environmental
impact, have received comparatively less attention  [1]. Factors such as
training time, prediction time and carbon footprint are also critical for
effectively using them in production. SaaS decision support systems use the
output of ML models to provide actionable recommendations, such as running
reactivation campaigns for users who are likely to churn. To this end, in this
paper, we present a benchmarking comparison of 17 different ML models for
churn prediction in SaaS, which include cloud deployment efficiency metrics
(e.g. latency, prediction time, etc.) and sustainability metrics (e.g., CO2
emissions, consumed energy, etc.) along with predictive performance metrics
(e.g., AUC, Log Loss, etc.). Two public datasets are employed, experiments are
repeated on four different machines, locally and on the Cloud, while a new
weighted Green Efficiency Weighted Score (GEWS) is introduced, towards
choosing the simpler, greener and most efficient ML model. Experimental
results indicated XGBoost and LightGBM as the models capable of offering a
good balance on predictive performance, fast training and inference times,
and limited emissions, while it was confirmed that the importance of region
selection towards minimizing the carbon footprint of the ML models.

## Instructions
The preprocessing is in two scripts, one for each dataset. The resulting results are the "preprocessed data".

We then use these as input to the training-evaluation which runs on different machines.

As it is, it reads the preprocessed data from the google cloud where we have uploaded them after we ran the preprocessed.

On the pc it reads them locally. (It is in the script commented).

For codecarbon it has a parameter for the region that we change according to the region of the google cloud vm.

The results are saved in csv files, one total for cross validation and one for the holdout (after the end of each training we save it with whatever results it has run so that it is lost).

## Citation

If you use this work, please cite:

```bibtex
@article{mlsaasbench2025,
  title   = {Beyond Accuracy: Benchmarking Machine Learning Models for Efficient and Sustainable SaaS Decision Support},
  author  = {Efthimia Mavridou, Eleni Vrochidou, Michail Selvesakis and George A. Papakostas},
  year    = {2025},
  journal = {Future Internet},
  publisher = {MDPI},
}
