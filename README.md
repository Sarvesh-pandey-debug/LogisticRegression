# LogisticRegression
Predicting if a person would buy life insurnace based on his age using logistic regression
Above is a binary logistic regression problem as there are only two possible outcomes (i.e. if person buys insurance or he/she doesn't).

import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
df = pd.read_csv("insurance_data.csv")
df.head()
age	bought_insurance
0	22	0
1	25	0
2	47	1
3	52	0
4	46	1
plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
<matplotlib.collections.PathCollection at 0x20a8cb15d30>

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)
X_test
age
4	46
8	62
26	23
17	58
24	50
25	54
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)
X_test
age
16	25
21	26
2	47
y_predicted = model.predict(X_test)
model.predict_proba(X_test)
array([[0.40569485, 0.59430515],
       [0.26002994, 0.73997006],
       [0.63939494, 0.36060506],
       [0.29321765, 0.70678235],
       [0.36637568, 0.63362432],
       [0.32875922, 0.67124078]])
model.score(X_test,y_test)
1.0
y_predicted
array([1, 1, 0, 1, 1, 1], dtype=int64)
X_test
age
4	46
8	62
26	23
17	58
24	50
25	54
model.coef_ indicates value of m in y=m*x + b equation

model.coef_
array([[0.04150133]])
model.intercept_ indicates value of b in y=m*x + b equation

model.intercept_
array([-1.52726963])
Lets defined sigmoid function now and do the math with hand

import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def prediction_function(age):
    z = 0.042 * age - 1.53 # 0.04150133 ~ 0.042 and -1.52726963 ~ -1.53
    y = sigmoid(z)
    return y
age = 35
prediction_function(age)
0.4850044983805899
0.485 is less than 0.5 which means person with 35 age will not buy insurance

age = 43
prediction_function(age)
0.568565299077705
0.485 is more than 0.5 which means person with 43 will buy the insurance

Exercise
Download employee retention dataset from here: https://www.kaggle.com/giripujar/hr-analytics.

Now do some exploratory data analysis to figure out which variables have direct and clear impact on employee retention (i.e. whether they leave the company or continue to work)
Plot bar charts showing impact of employee salaries on retention
Plot bar charts showing corelation between department and employee retention
Now build logistic regression model using variables that were narrowed down in step 1
Measure the accuracy of the model
