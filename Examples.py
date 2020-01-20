# How to use TimeBasedCV
data_for_modeling=pd.read_csv('data.csv', parse_dates=['record_date'])
tscv = TimeBasedCV(train_period=30,
                   test_period=7,
                   freq='days')
for train_index, test_index in tscv.split(data_for_modeling,
                   validation_split_date=datetime.date(2019,2,1), date_column='record_date'):
    continue

# get number of splits
tscv.get_n_splits()

#### Example- compute average test sets score: ####
X = data_for_modeling[['record_date',columns]]
y = data_for_modeling[label]
from sklearn.linear_model import LinearRegression
import numpy as np

scores = []
for train_index, test_index in tscv.split(X, validation_split_date=datetime.date(2019,2,1)):

    data_train   = X.loc[train_index].drop('record_date', axis=1)
    target_train = y.loc[train_index]

    data_test    = X.loc[test_index].drop('record_date', axis=1)
    target_test  = y.loc[test_index]

    # if needed, do preprocessing here

    clf = LinearRegression()
    clf.fit(data_train,target_train)

    preds = clf.predict(data_test)

    # accuracy for the current fold only    
    r2score = clf.score(data_test,target_test)

    scores.append(r2score)

# this is the average accuracy over all folds
average_r2score = np.mean(scores)
#### End of example ####

#### Example- RandomizedSearchCV ####
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor
from random import randint, uniform

tscv = TimeBasedCV(train_period=10, test_period=3)
index_output = tscv.split(data_for_modeling, validation_split_date=datetime.date(2019,2,1))

lgbm = LGBMRegressor()

lgbmPd = {" max_depth": [-1,2]
         }

model = RandomizedSearchCV(
    estimator = lgbm,
    param_distributions = lgbmPd,
    n_iter = 10,
    n_jobs = -1,
    iid = True,
    cv = index_output,
    verbose=5,
    pre_dispatch='2*n_jobs',
    random_state = None,
    return_train_score = True)

model.fit(X.drop('record_date', axis=1),y)
model.cv_results_
#### End of example ####
