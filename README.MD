## Geotag Predictor

This is the code for predicting geotag of tweets.

### Dataset
See [Eisenstein, Jacob, et al.](http://www.cs.cmu.edu/~nasmith/papers/eisenstein+oconnor+smith+xing.emnlp10.pdf)
  Although the single file size of datasets is greater than 50.00 MB (See, [http://git.io/iEPt8g ](http://git.io/iEPt8g )), they were added in datasets for convenience.

### Requirements
+ python3+

#### Installation
```pip
pip install -r requirements.txt
```
### Usage
#### Train
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
#### Predict
```python
python run.py -p models/ datasets/dev-best200.csv 

```
#### Score
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
#### Train&Predict&Score
```python
python run.py \
    -t datasets/train-best200.csv datasets/dev-best200.csv \
    -p models/ datasets/dev-best200.csv \
    -s results/final_results.csv  datasets/dev-best200.csv 
```

#### Others
```python
python run.py -h
```

### Used library
+ [sklearn](https://scikit-learn.org/stable/index.html)
+ [pandas](https://github.com/pandas-dev/pandas.git)
+ [tqdm](https://github.com/tqdm/tqdm.git)
### License
See [LICENSE](LICENSE) file.