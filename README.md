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
    
    
    
### heatmap

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

## date-time

    df1['Start'] = df1['Start'].dt.strftime('%b %d %Y %X')
    df1['Finish'] = df1['Finish'].dt.strftime('%b %d %Y %X')
    
    
    df2['FromTime'] = pd.to_datetime(df2['FromTime'])
    df2['ToTime'] = pd.to_datetime(df2['ToTime'])
    
    
    from datetime import datetime, date, timedelta

    import datetime
    start_datetime= datetime.datetime(2022,5,11, 1,29,0)

    print(start_datetime + timedelta(minutes = 735 ))
    
    
    df['spend_date_mt']=pd.to_datetime(df['spend_date_mt'].astype(str), format='%d-%m-%Y')
    
    
    no_of_day=[]
    for i in range(len(df)):
        e_start_date=df.e_start_date[i]
        prevopenddate=df.prevopenddate[i]
        d = e_start_date - prevopenddate
        no_of_day.append(d)
    df["no_of_days"]= no_of_day 
    df['difference'] = df['no_of_days'] / pd.Timedelta('1 hours')
    
    
    
###  Time-Series

    ## resample abbrebation

    B         business day frequency
    C         custom business day frequency (experimental)
    D         calendar day frequency
    W         weekly frequency
    M         month end frequency
    SM        semi-month end frequency (15th and end of month)
    BM        business month end frequency
    CBM       custom business month end frequency
    MS        month start frequency
    SMS       semi-month start frequency (1st and 15th)
    BMS       business month start frequency
    CBMS      custom business month start frequency
    Q         quarter end frequency
    BQ        business quarter endfrequency
    QS        quarter start frequency
    BQS       business quarter start frequency
    A         year end frequency
    BA, BY    business year end frequency
    AS, YS    year start frequency
    BAS, BYS  business year start frequency
    BH        business hour frequency
    H         hourly frequency
    T, min    minutely frequency
    S         secondly frequency
    L, ms     milliseconds
    U, us     microseconds
    N         nanoseconds


### other commands
        
        
    [df.loc[df['tag'] == i, 'links'] for i in all_links]
    
    [df.loc[df["tag"]==i].links.values[0] for i in l]
    
    df.drop(df[df['Age'] < 25].index, inplace = True)
    
    df.rename(columns=lambda x: x.replace('$', ''), inplace=True)
    
    df.loc[df['id'] == 20 ], 'budget'] = 50
    
    df.loc[df['release_date'].isnull() == true ], 'release_date'] = '01/01/1998'
    
    with open('dict.pkl', 'wb') as f:
        pickle.dump(d, f)
        
        
    with open('dict.pkl', 'rb') as f:
        d = pickle.load(f)
        
        
        
    import plotly.express as px
        
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,'percent_missing': percent_missing})
    missing_value_df.sort_values(by=['percent_missing'], inplace=True,ascending=False )
    fig = px.bar(missing_value_df, x='column_name', y='percent_missing', title="Missing value %",height=700)
    fig.show()
        
        
        
   
    
    df1['rel'] = df1['rel'].apply(lambda x: '(' + x + ')' if x != " " else "" )
    
    df1['order_no'] = df1[['order_no','rel']].apply(lambda x: ''.join(map(str,x)), axis=1)
    
    df1["rel"] = ["Yes" if len(df1.rel[i]) == 0 else "NO" for i in range(len(df1))]
    
    df1['order_no'] = df1[['order_no','rel']].apply(lambda x: ''.join(map(str,x[x.notnull()])), axis=1)
    
    df1['new_col'] = df1['rem_ops'].apply(lambda x: "yes" if 'A' in str(x) else "" ) 
    
    df1['prev_op_end_time'].mask(df1['prev_op_end_time'] == 'First Operation', '', inplace=True)
    
    ignore_name =['CCLK','No-Plastic','NoWash']
    df['special_column'] = df['Special'].apply(lambda x: ','.join(list(set(x.split(",")) - set(ignore_name))))
    
    from collections import Counter
    Counter(train_copy.dtypes.values)
    
    
    index_names = df[ (df.Device == "Montage Endtaetigkeit / stat. IBS (Werk)") | (df.Device == "Montage")].index
    df.drop(index_names, inplace = True)
    df.reset_index(inplace=True)
    
    df1 = df[(df["Production_line"] == pl_id )  & (df["Task"] == "Operation Change")]
    df1 = df[(df["Production_line"].notnull() )  & (df["Task"] == "Operation Change")]
    
    
    
    df["Detail"] = df["Order_Op"].apply(lambda x: x.split("_")[0] if str(x)!="None" else "")  -> 256_AA1 to 256
    
    df1['rel_letter'] = df_slitter['rel_letter'].apply(lambda x: '(' + x + ')' if x != " " else "" )
    df1['order_no'] = df_slitter[['order_no','rel_letter']].apply(lambda x: ''.join(map(str,x)), axis=1)
    df1['prev_op'] = df1[['prev_op_type','prev_op_end_time']].apply(lambda x: ' '.join(map(str,x[x.notnull()])), axis=1)
    df1['rel_due_date'] = df1[['rel_due_date','HoldCode']].apply(lambda x: '/'.join(map(str,x[x.notnull()])), axis=1)
    
    df1=df_slitter.astype({"CoreCode":str})
    
    df1['CoreCode'] = df1['CoreCode'].apply(lambda x: x[:2] if 'HD Fiber' in str(x) else x[:1] if str(x)!="nan" else "")
    df1['cc']= df1['machine_id'].apply(lambda x: int(str(x.replace('F',""))))
    
    
    naive[naive['e_start_date'].notna()].shape
    
    df = df[df['A'].isin([3,6])]  , filter row for value having 3,6
    
    df = df[df.col_name != value].reset_index(drop=True)
    
    xls = pd.ExcelFile('path\\file_name.xlsx')
    df1 = pd.read_excel(xls, 'sheet_name')
    
    df3 = pd.concat([df1, df2], ignore_index = True)
    
    
    df = penalty_df.copy()
    df1_grouped  = df.groupby(['col1', 'col2'])
    count = 0

    for group_name, df_group in df1_grouped:
        if df_group.shape[0] == 1:
            continue
    
    
    
    
