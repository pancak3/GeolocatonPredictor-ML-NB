
# Geotag Predictor  
This is the code for predicting geotag of tweets trainning on token frequencies using Decision Tree and Naïve Bayes.  
## Implementation  
###  Feature selection
In `util/preprocessing/merge.py`,  
+ `feature_filter` shows it drops single character features like `[a, b, ..., n]`  
- `merge` shows it intuitively merges similar features like `[aha, ahah, ..., ahahahaha]` and `[taco, tacos]`  
+ `merge` also shows it uses [Recursive Feature Elimination](https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#sphx-glr-auto-examples-feature-selection-plot-rfe-digits-py) to rank features and select best 300 features.  
###  Classifier Combination 
In `preprocess/merge.py`,  
+ `result_combination` shows it uses results of [Decision Tree](https://scikit-learn.org/stable/modules/tree.html), [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [Bernoulli Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)), [Complement Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html) and [Multinomial Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) to vote out the majority predictions while the following shows the models can be slightly different every time training.  
#### Instance manipulation
In `util/train.py`,  
+ `complement_nb` shows it uses [bagging](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) to generate multiple training datasets.   
+ `complement_nb` also shows it uses [42-Fold Cross Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to generate multiple training datasets.   
  
#### Algorithm manipulation
In `util/train.py`,   
+ `complement_nb` also shows it uses [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to generate multiple classifiers and select the best based on `accuracy`.   
### Dataset  
+ See [Eisenstein, Jacob, et al.](http://www.cs.cmu.edu/~nasmith/papers/eisenstein+oconnor+smith+xing.emnlp10.pdf)  
  >Although some file's size in datasets is greater than 50.00 MB, it was still added in `datasets` for convenience. (See, [http://git.io/iEPt8g ](http://git.io/iEPt8g ))  
  
## Requirements  
+ python3+  
```pip  
pip install -r requirements.txt  
```  
## Usage  
> **Note:** The code will remove the old models and results every time running.  <strong>MAKE SURE</strong> you have saved your satisfying models..### Train  
```python  
python run.py -t datasets/train-best200.csv datasets/dev-best200.csv  
  
```  
the output would be like:  
```  
INFO:root:[*] Merging datasets/train-best200.csv   
 42%|████████         | 1006/2396 [00:05<00:20, 92.03 users/s]  
...  
...  
[*] Saved models/0.8126_2019-10-02_20:02  
[*] Accuracy: 0.8125955095803455  
 precision    recall   f_scoreCalifornia   0.618944  0.835128  0.710966  
NewYork      0.899371  0.854647  0.876439  
Georgia      0.788070  0.622080  0.695305  
weighted     0.827448  0.812596  0.814974  
```  
### Predict  
```python  
python run.py -p models/ datasets/dev-best200.csv   
```  
the output would be like:  
```  
...  
INFO:root:[*] Saved results/final_results.csv  
INFO:root:[*] Time costs in seconds:  
 PredictTime_cost  11.98s  
```  
### Score  
```python  
python run.py -s results/final_results.csv  datasets/dev-best200.csv  
```  
the output would be like:  
```  
[*] Accuracy: 0.8224697308099213  
 precision    recall   f_scoreCalifornia   0.653035  0.852199  0.739441  
NewYork      0.747993  0.647940  0.694381  
Georgia      0.909456  0.858296  0.883136  
weighted     0.833854  0.822470  0.824577  
INFO:root:[*] Time costs in seconds:  
 ScoreTime_cost  1.48s  
  
```  
### Train&Predict&Score  
```python  
python run.py \  
 -t datasets/train-best200.csv datasets/dev-best200.csv \ 
 -p models/ datasets/dev-best200.csv \ 
 -s results/final_results.csv datasets/dev-best200.csv 
```  
  
### Help  
```python  
python run.py -h  
```  
  
## Used libraries  
+ [sklearn](https://scikit-learn.org/stable/index.html) for easily using Complement Naive Bayes, some feature selectors and other learning tools.  
+ [pandas](https://github.com/pandas-dev/pandas.git)[numpy](https://github.com/numpy/numpy.git)  for easily handling data.  
+ [tqdm](https://github.com/tqdm/tqdm.git) for showing the process of loop.  
+ [joblib](https://joblib.readthedocs.io/en/latest/) for dumping/loading memory to/from disk.  
+ <strike>[nltk](https://github.com/tqdm/tqdm.git) for capturing word types on the purpose of feature filtering</strike>  
## License  
See [LICENSE](LICENSE) file.
