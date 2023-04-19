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

![datasetInfo](https://user-images.githubusercontent.com/72138112/233052550-a3adf737-2f53-4e16-8e05-fa5e99df133d.jpg)

_Final Results of optimization_

|Sample	 |   Best Accuracy  |   Best Kernel   |   Best Nu  |   Best Epsilon  |
|------  | -------------    | --------------  | -----------|---------------  |
|1       |	0.57        |      poly       |    3.37	   |       4.79      |
|2	 |      0.86	    |      linear     |	   5.09	   |       7.15      |
|3	 |       0.82	    |      rbf	      |     2.55  |      8.86      |
|4	 |       0.78	    |      sigmoid    |	   2.78	   |       8.50      |
|5	 |       0.94	    |      linear     |	   0.43	   |       0.68      |
|6	 |       0.92	    |      rbf	      |     6.81   |	   0.46      |
|7	 |       0.46	    |      poly	      |     5.98   |       0.38      |
|8	 |       0.58	    |      poly	      |     6.95   |	   5.38      |
|9	 |       0.83	    |      rbf	      |     2.42   |       5.35      |
|10	 |       0.83	    |      poly	      |     2.17   |	   9.78      |




