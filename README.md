# Natural-Language-Processing

All files are written in Python 3.8

## Packages Used
- numpy
- nltk

## nltk data
- punkt
- brown
- universal_tagset

### Running HMM
- python HMM.py 
- Takes around 2 minutes for 5-fold cross validation


### Running SVM
- Folder contains 2 versions
- In SVM.py model is being trained on 1/5 th of total dataset but with more features whereas SVM2.py 's model is being trained on total dataset but with less no of features due to memory constraints
- SVM.py gives better accuracy than 
- Used Colab for running as it takes more thas 8 GB of ram 

### Running BiLSTM
- Used Glove embeddings
- Used Keras for Neural Network
- Glove folder must be placed in the BiLSTM directory before running
- Command : python HMM.py
