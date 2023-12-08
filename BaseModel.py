import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

"""
Input variables are the following: 
Country : land (Categorical)
Year : year (numeric)
democ : how democratic the country is (numeric)
autoc : how autocratic the country is (numeric)
polity : combined score of autoc and democ with a range of -10 to 10 (numeric)
polity2 : revised polity score
durable : Shows the duration of a regime untill a regime change happens (numeric)
xrreg : 
xrcomp
xropen
xconst
parreg
parcomp
exconst
d5 : Shows if a regime change happened in that year (boolean)
sf : Flag when a state failure happened (boolean)
regtrans : an indicator how heavy the regime shift was with an range of -2 to +3 with some special auxiliary codes (numeric)
"""