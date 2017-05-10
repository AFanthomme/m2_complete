# Stage m2 : higgs generation mechanism classification


#### Using Monte-Carlo simulated data, train neural network classifiers, generate content plots and "pickle" them for later use (note that we remove the "fit" method before storing so that our predictor cannot be altered).  

- This is the development version with all the details / functions / controls and lack of some optimizations (reload datasets everytime you train a new model

- Efficient version for meta-parameter exploration on cluster incoming at https://github.com/AFanthomme/m2_stand_alone


## To do :
- try batched SVC
- try kernels for logreg with 
- generate the signal strength for real data


## Notes :
- Most of the program's behaviour can be controlled from constants.py 
- The intended use is to launch directly plotter.py. It will try to generate and save in saves/tmp the category contents for all the classifiers defined in models_dict (Note that it is lazy and will not recompute a model if one with the same name has already been trained, however it is not computationally optimized.)

## Best models so far :
With discriminants :
- logistic regression with invfreq weights 
- bag_tree invfreq (if we only care about VBF)
- 

Without discriminants :
- adaboost log reg (but very bad on VH)


