# Parameter-Optimization-of-SVM-102003742_Arjun Pundir
## Optimization of the SVM parameters for 10 different samples with 1000 iterations to get the best parameters.

Use the link to get the datasets from UCI Machine learning Repository--->[link here](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+)

## DataSet Used
Experimental data used for binary classification (room occupancy) from Temperature,Humidity,Light and CO2. Ground-truth occupancy was obtained from time stamped pictures that were taken every minute.

## Previosuly install Libraries 
```
bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')
```

## colab File Upload with Results

![Dataset Info](./C:/Users/MY/OneDrive/Desktop/datasetinfo.jpeg)


