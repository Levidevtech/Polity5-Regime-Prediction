# Polity5-Regime-Prediction

Requirements:
- Python 3.6 or higher
- Pip

To run the code you need the following packages:
- pandas ( pip install pandas )
- sklearn ( pip install scikit-learn )
- numpy ( pip install numpy )
- matplotlib ( pip install matplotlib )
- xlrd ( pip install xlrd )
- seaborn ( pip install seaborn )
- statsmodels ( pip install statsmodels )

Either use python or python3 depending on your system.

To run the code, open a terminal and navigate to the folder where the code is located. Then run the following command:
```python RandomForest.py```

The code will then run and print the results to the terminal. If a country code is provided, a graph will be shown with the predicted and actual values for that country like:
```python RandomForest.py --country=210```

If you want to train the model on a specific country, you can use the following command:
```python RandomForest.py --ccode=210```

If you want to see the decision tree, you can use the following command with the max depth of the tree:
```python RandomForest.py --depth=3```


              precision    recall  f1-score   support

          -2       0.75      0.39      0.51       128
          -1       0.44      0.19      0.27        21
           0       0.95      0.99      0.97      4869
           1       0.46      0.28      0.35       117
           2       0.62      0.49      0.55       227

    accuracy                           0.93      5362
   macro avg       0.64      0.47      0.53      5362
weighted avg       0.92      0.93      0.92      5362

Example of a print out from random forest.


Logistic regression:

To run:
```python BaseModelOfLogistic.py```




