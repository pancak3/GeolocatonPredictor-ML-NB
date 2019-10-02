# Geotag Predictor
This is the code for predicting geotag of tweets based on Complement Naive Bayes.
## Implementation
###  Feature selection 
In `util/preprocessing/merge.py`,
+  the method `feature_filter` shows that it drops single charachter features like `[a, b, ..., n]`
-  the method `merge` shows that it intuitively merges the similar features frequncy like `[aha, ahah, ..., ahahahaha]` and `[taco, tacos]`
+  the method `merge` also shows that it uses [Chi-square](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html) with a threshold `0.001` to select features that are more realtive with classes.
###  Classifier Combination 
In `preprocess/merge.py`,
+ the method `result_combination` shows that it uses multiple results to vote out the majority prediction while the following shows that the models can be slightly different every time training.
#### Instance manipulation
In `util/train.py`,
+ the method `complement_nb` shows that it uses [bagging](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) to generate multiple training datasets. 
+ the method `complement_nb` also shows that it uses [42-Fold Cross Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to generate multiple training datasets. 

#### Algorithm manipulation:
In `util/train.py`, 
+ the method `complement_nb` also shows that it uses [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to generate multiple classifiers. 
### Dataset
+ See [Eisenstein, Jacob, et al.](http://www.cs.cmu.edu/~nasmith/papers/eisenstein+oconnor+smith+xing.emnlp10.pdf)
  Although some file's size in datasets are greater than 50.00 MB , they were added in datasets for some convenience. (See, [http://git.io/iEPt8g ](http://git.io/iEPt8g ))

## Requirements
+ python3+

## Installation
```pip
pip install -r requirements.txt
```
## Usage
### Train
```python
python run.py -t datasets/train-best200.csv datasets/dev-best200.csv

```
the output would be like:
```
INFO:root:[*] Merging datasets/train-best200.csv 
 22%|█████          | 539/2396 [00:05<00:20, 92.03 users/s]
...
...
[*] Saved models/0.8126_2019-10-02_20:02
[*] Accuracy: 0.8125955095803455
            precision    recall   f_score
California   0.618944  0.835128  0.710966
NewYork      0.899371  0.854647  0.876439
Georgia      0.788070  0.622080  0.695305
weighted     0.827448  0.812596  0.814974
```
### Predict
```python
python run.py -p models/ datasets/dev-best200.csv 

```
### Score
```python
python run.py -s results/final_results.csv  datasets/dev-best200.csv
```
the output would be like:
```
[*] Accuracy: 0.8224697308099213
            precision    recall   f_score
California   0.653035  0.852199  0.739441
NewYork      0.747993  0.647940  0.694381
Georgia      0.909456  0.858296  0.883136
weighted     0.833854  0.822470  0.824577
INFO:root:[*] Time costs in seconds:
              Score
Time_cost  1.478792

```
### Train&Predict&Score
```python
python run.py \
    -t datasets/train-best200.csv datasets/dev-best200.csv \
    -p models/ datasets/dev-best200.csv \
    -s results/final_results.csv  datasets/dev-best200.csv 
```

### Others
```python
python run.py -h
```

## Used library
+ [sklearn](https://scikit-learn.org/stable/index.html) for easily using Complement Naive Bayes, some feature selectors and other learning tools.
+ [pandas](https://github.com/pandas-dev/pandas.git) for easily handling data
+ [tqdm](https://github.com/tqdm/tqdm.git) for showing the process of loop
## License
See [LICENSE](LICENSE) file.