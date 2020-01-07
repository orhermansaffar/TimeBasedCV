import pandas as pd
import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *

class TimeBasedCV(object):
    '''
    Parameters 
    ----------
    train_period_days: int
        number of days to include in each train set 
    test_period_days: int
        number of days to include in each test set 
    '''
    
    
    def __init__(self, train_period_days=30, test_period_days=7):
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days 

        
        
    def split(self, data, validation_split_date=None):
        '''
        Parameters 
        ----------

        Data: your data, contain one column for the record date named "record_date" 
        validation_split_date: first date to perform the splitting on.
            if not provided will set to be the minimum date in the data after the first training set

        Returns 
        -------
        train_index ,test_index: 
            index for train and test set similar to sklearn model selection
        '''
        train_indices_list = []
        test_indices_list = []

        if validation_split_date==None:
            validation_split_date = data['record_date'].min().date() + relativedelta(days=self.train_period_days)

        start_train = validation_split_date - relativedelta(days=self.train_period_days)
        end_train = start_train + relativedelta(days=self.train_period_days)
        start_test = end_train
        end_test = start_test + relativedelta(days=self.test_period_days)

        while end_test < data['record_date'].max().date():
            # train indices:
            cur_train_indices = list(data[(data['record_date'].dt.date>=start_train) & 
                                     (data['record_date'].dt.date<end_train)].index)

            # test indices:
            cur_test_indices = list(data[(data['record_date'].dt.date>=start_test) &
                                    (data['record_date'].dt.date<end_test)].index)

            train_indices_list.append(cur_train_indices)
            test_indices_list.append(cur_test_indices)

            # update dates:
            start_train = start_train + relativedelta(days=self.test_period_days)
            end_train = start_train + relativedelta(days=self.train_period_days)
            start_test = end_train
            end_test = start_test + relativedelta(days=self.test_period_days)

        # mimic sklearn output  
        index_output = [(train,test) for train,test in zip(train_indices_list,test_indices_list)]

        return index_output

# How to use TimeBasedCV
data_for_modeling=pd.read_csv('data.csv')
data_for_modeling['record_date']=data_for_modeling['record_date'].values.astype('datetime64[D]')
tscv = TimeBasedCV(train_period_days=10,
                   test_period_days=3)
for train_index, test_index in tscv.split(data_for_modeling,
                   validation_split_date=datetime.date(2019,2,1)):
    print(train_index)
    print(test_index)
