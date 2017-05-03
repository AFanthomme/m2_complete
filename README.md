# Stage m2 : higgs generation mechanism classification


Using Monte-Carlo simulated data, train neural network classifiers, generate content plots and "pickle" them for later use (note that we remove the "fit" method before storing so that our predictor cannot be altered).  

# To do :
- create new categories based on VH_lept / VH_Hadr / ... instead of simple merged VH
- try random forests, batched SVC
- automate weights exploration
- generate the signal strength figure using the real data instead of simulated one


# Notes :
- Most of the program's behaviour can (and should) be modified by only touching constants.py 
- The intended use is to launch directly plotter.py. It will try to generate and save in saves/tmp the category contents for all the classifiers defined in models_dict (Note that it is lazy and will not recompute a model if one with the same name has already been trained)

- linear logistic regression with invfreq weights achieves similar results as what was done previously 
- log_reg without kinematic discriminants and without weights is almost as good as with discr.
- MLP classifiers perform very strangely, especially when the computed discriminants are removed


