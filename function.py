import pandas as pd
import statistics
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pandas.api.types import is_numeric_dtype
import numpy as np
from IPython.display import display
from statistics import mode, mean



class Utils:


    # make delta time column from timestamps
    def dt(df):
        list_ = []
        for i in range(len(df)-1):
            if df.iloc[i+1]["caseid"] == df.iloc[i]["caseid"]:
                list_ = list_ + [df.iloc[i+1]["ts"]-df.iloc[i]["ts"] ]
            else:
                list_ = list_ + [list_[-1]]
        list_ = list_ + [list_[-1]]

        return list_
    

    # calculate the percentage of nan values of each column of the dataset
    def persantage_nan(df):
        return ((df.isna().sum())/len(df))*100


    # predict nan values for the column using RandomForest
    def predict_null_value(colum,df):
    
        df_to_encode = []

        df_t = df.copy()
        
        # take out the columns to predict
        y = df_t.pop(colum)
        
        
        # columns with non Nan value
        columns = df_t.columns[[i for i in df_t.isna().sum()==0]]
        df_t = df_t[columns]
        
        
        # take the value that are nan
        nan = y[y.isna()].index
        
        
        # convert to numeric the columns that are possibile
        for column in columns:
            try:
                df_t[column] = pd.to_numeric(df_t[column])
            except:
                df_to_encode.append(column)
        
        # encode the columns that are not numeric
        df_encoded = pd.get_dummies(df_t, columns=df_to_encode)
        
        
        # take out the train df which is the one with no null
        X_train = df_encoded.drop(nan)
        Y_train = y.drop(nan)
        
        X_test = df_encoded.drop(y[[not elem for elem in y.isna()]].index)
        
        
        # train a calssifier or a Regression
        if is_numeric_dtype(y):
        
            model = RandomForestRegression(n_jobs=-1)
        else:
            model = RandomForestClassifier(n_jobs=-1)
            
        model.fit(X_train[:1000], Y_train[:1000])
        
        # make prediction and change the columns
        new_col = []
        prediction = model.predict(X_test)
        i=0
        for el in df[colum]:
            if el is np.nan:
                new_col.append(prediction[i]) 
                i+=1
            else:
                new_col.append(el)
        
        
        df[f"{colum}_was_null"] = df[colum].isna()
        df[colum] = new_col
    
        return df, model


    # transform list value into a single value after groupby
    def reduce_list_columns(df):
        # take out the index
        try:
            df = df.drop(["index"],axis=1)
        except:
            pass
    
        # for each element of the column we check if the list has all the same value, if that is the case we just pust the first value
        new_df = df.copy()
        for col in df.columns:
            x = new_df[col].iloc[0]
            if(type(x) == list):
                control=0
                for i in range(len(df)):
                    x = new_df[col].iloc[i]
                    if len(set(x)) != 1:
                        control = 1
                if control ==0:
                    new_df[col] = new_df[col].apply(lambda x: x[0])
        return new_df


    # show the columns that could be used for agregation
    def display_columns_to_aggragate(df_grouped):
        c = []
        for col in df_grouped.columns:
            x = df_grouped[col].iloc[0]
            if(type(x) == list):
                c.append(col)
        display(df_grouped[c])
        
        
    # perform aggregation encoding 
    def aggregation_encoding(df_grouped,df):

        df_grouped_t = df_grouped.copy()

        for col in df_grouped.columns:    
            #the columns need to have only list elements
            x = df_grouped[col].iloc[0]

            if (col != "dt"):
                if(type(x) == list):
                    #if there is a float inside the list we should make avg,max,min
                    if type(x[0]) == float:
                        df_grouped_t[col + "_avg"] = df_grouped_t[col].map(lambda x : mean(x))
                        df_grouped_t[col + "_max"] = df_grouped_t[col].map(lambda x : max(x))
                        df_grouped_t[col + "_min"] = df_grouped_t[col].map(lambda x : min(x))
                        df_grouped_t.drop(col,axis=1,inplace = True)

                    #if there is an int we do mode inside the list we should make mode
                    if type(x[0]) == int:
                        df_grouped_t[col + "_mode"] = df_grouped_t[col].map(lambda x : mode(x))
                        df_grouped_t.drop(col,axis=1,inplace = True)
                    
                    if type(col) not in [float, int, pd._libs.tslibs.timedeltas.Timedelta]:
                        for el in df[col].unique():
                            df_grouped_t[el] = df_grouped_t[col].map(lambda x : x.count(el)/len(x)) # frequency of an element
                        df_grouped_t.drop(col,axis=1, inplace = True)

        df_grouped_t["avg_dt"] = df_grouped_t["dt"].map(lambda x : np.mean(x))
        df_grouped_t["max_dt"] = df_grouped_t["dt"].map(lambda x : np.max(x))
        df_grouped_t["min_dt"] = df_grouped_t["dt"].map(lambda x : np.min(x))

        df_grouped_t.drop("dt", axis=1, inplace = True)

        return df_grouped_t


    # keep only the columns that pass the nan threshold 
    def prod_nan_with_treshold(df,treshold=50):
        
        def persantage_nan(df):
            return ((df.isna().sum())/len(df))*100
    
        columns = df.columns[[i for i in persantage_nan(df)<=treshold]]
        df = df[columns]
        
        return df
    

    # drop the columns that are not valuable for the model training (like ID variables)
    def drop_id_columns(df, drop_id_threshold):
        df_new = df.copy()

        for col in df.columns:
            freq_value = len(df[col].unique())/df.shape[0]

            if (df[col].dtype != '<m8[ns]'):
                if (freq_value >= drop_id_threshold):
                    print("Dropping column " + col + " with value: " + str(freq_value))
                    df_new = df_new.drop([col], axis=1)
                
        return df_new


    # index encode columns from the list columns_to_encode
    def index_encoding_2(df_grouped, df, columns_to_encode):
        
        for column_to_encode in columns_to_encode:
            max_lenght = df_grouped[column_to_encode].map(lambda x: len(x)).max()
            
            for i in range(0, max_lenght):
                for act in df[column_to_encode].unique():
                    df_grouped[f"{act}_{i+1}"] = df_grouped[column_to_encode].map(lambda x : 1 if act in x else 0).copy()
                    
                t_i = []
            
                for index in range(0,len(df_grouped)):
                    try:
                        t_i.append(df_grouped.iloc[index]["ts"][i])
                    except:
                        t_i.append(0)
                
                df_grouped[f"t_{i+1}"] = t_i
        df_grouped.drop(columns_to_encode,inplace=True,axis=1)
        
        return df_grouped


    def index_encoding(df_grouped, df, columns_to_encode):
        lenghts = []
        max_lenght = df_grouped["activity"].map(lambda x: len(x)).max()

        if max_lenght>20:
            max_lenght = df_grouped["activity"].map(lambda x: len(x)).mode()[0]
            
        for i in range(0, max_lenght):
            for column_to_encode in columns_to_encode:
                for act in df[column_to_encode].unique():
                    df_grouped[f"{act}_{i+1}"] = df_grouped[column_to_encode].map(lambda x : 1 if act in x else 0).copy()
                    
                t_i = []
            
                for index in range(0,len(df_grouped)):
                    try:
                        t_i.append(df_grouped.iloc[index]["ts"][i])
                    except:
                        t_i.append(0)
                
                df_grouped[f"t_{i+1}"] = t_i
        df_grouped.drop(columns_to_encode,inplace=True,axis=1)
        
        return df_grouped