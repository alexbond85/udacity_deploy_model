# Model Card

## Model Details
Random Forest Classifier with no hyperparameter tuning.  

## Intended Use
Educational purposes. Demonstrates the possibility to train, serialize and deserialize 
a model.

## Training Data
Obtained through a random train-test split with the proportions 80%/20%.
The dataset is taken from https://archive.ics.uci.edu/ml/datasets/census+income. 

## Evaluation Data
Test data (20%) taken from the train/test split.  

## Metrics
Metrics of interest: precision, recall, fbeta. Evaluations of the trained model
on the test set:
precision=0.725, recall=0.634, fbeta=0.677

## Ethical Considerations
Model might exhibit bayes with respect to education, sex, marital status, race. 

## Caveats and Recommendations
In a real scenario the model should be carefully evaluated on the absence of
bayes described in the Ethical Considerations. It is very likely that it has 
to be retrained on the data with different sampling.