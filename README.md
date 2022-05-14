# ML-Commands


### Import
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None) 
    from pylab import rcParams
    rcParams['figure.figsize']= 20, 5
    
    from sklearn.preprocessing import StandardScaler

    from sklearn.metrics import mean_squared_error

    from sklearn.linear_model import LinearRegression

    from sklearn.tree import DecisionTreeClassifier

    from sklearn.ensemble import AdaBoostClassifier

    from sklearn.ensemble import GradientBoostingRegressor

    from sklearn.decomposition import PCA

    from sklearn.model_selection import GridSearchCV
    
    
### get all categorial columns name
    df.select_dtypes(include=['object']).columns.tolist()
   
### get all numerical columns name
    df.select_dtypes(include=np.number).columns.tolist()
    
    
### for confusion matrix

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_predicted)
    cm
    %matplotlib inline
    import matplotlib.pyplot as plt
    import seaborn as sn
    plt.figure(figsize=(10,7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    
### for doing one_hot encoding fot all categorical features

    def category_onehot_categorical_col(categorical_col,final_df):
    df_final=final_df
    i=0
    for fields in categorical_col:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final
    
    
    
### other commands

    num_df = df.select_dtypes(include=np.number)
   
    corr_df=num_df.corr()
    plt.figure(figsize=(16,9))
    sns.heatmap(corr_df, annot=True, linewidths=2)
    
    
    
    
    # with the following function we can select highly correlated features
    # it will remove the first feature that is correlated with anything other feature

    def correlation(dataset, threshold):
        col_corr = set()  # Set of all the names of correlated columns
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        return col_corr
