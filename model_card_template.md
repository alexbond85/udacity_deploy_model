# Model Card

## Model Details
Random Forest Classifier with no hyperparameter tuning.  

## Intended Use
Educational purposes. Demonstrates the possibility to train, serialize and deserialize 
a model. As well as performing metrics computation.

## Training Data
Obtained through a random train-test split with the proportions 80%/20%.
The dataset is taken from https://archive.ics.uci.edu/ml/datasets/census+income. 

## Evaluation Data
Test data (20%) taken from the train/test split.  

## Metrics
Evaluations on the test set: 
precision=0.725, test_recall=0.634, fbeta=0.677

## Ethical Considerations
Not applicable.

## Caveats and Recommendations
Nothing to mention.