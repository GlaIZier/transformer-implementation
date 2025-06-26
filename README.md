# Transformer implementation

### Data
Training data is taken from:
https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench

### Results
Couldn't train a large model on the whole dataset due to limited resources. It's also hard for the model to grasp the language grammar by training on such a small and not very consistent dataset.
Use the default params for training/inference and the small dataset to train and test the model. 
Transformer works with both tiktokenizer and my own minbpe implementation.