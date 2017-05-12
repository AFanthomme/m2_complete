# Stage m2 : higgs generation mechanism classification


#### Using Monte-Carlo simulated data, train neural network classifiers, "pickle" them for later useg and enerate content plots.  
## The whole program is controlled by the main.py script, calling the functions from the src package :
- src.constants regroups all the program's global variables.
- src.preprocessing generates .txt saves of the datasets with some control over the features to retrieve / compute / remove.
- src.trainer trains the specified classifier and stores it in an appropriate folder along with its predictions.
- src.plotter takes a trained model, a test set, and generates the associated category content plot.

- In progress : SelfThresholdingAdaClassifier provides a wrapper of sklearn's adaboost meta-estimator designed to provide more control over its prediction method (especially optimize some prediction thresholds to minimize a given cost).

