## ML-Commands


---

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
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=42)
    
    
---

    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    X_scaled=scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)
    
    
---
### Profiling Results

    from pandas_profiling import ProfileReport
    from IPython.display import display, HTML
    df_profile = ProfileReport(df, minimal=True, title="Profiling Report", progress_bar=False, infer_dtypes=False)
    profile_html = df_profile.to_html()


    display(HTML(profile_html))



---


### Confusion Matrix

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
    
### One-Hot Encoding

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
    

---
    
### Heatmap

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


---

## Date-Time

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
    
    
    def time_mapping(h):
    if 8<h<12:
        return 'Morning'
    
    elif 12<h<16:
        return "Afternoon"
    else:
        return "Evening"
    
    
    df['Time'] = df['Time'].apply(lambda x: time_mapping(int(x.strftime("%H"))))
    
    #remove microsec from datetime col
    df['datetime_col'] = df['datetime_col'].dt.floor('1s')
    

---
    
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


---

### Plotly

    import plotly.express as px
        
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,'percent_missing': percent_missing})
    missing_value_df.sort_values(by=['percent_missing'], inplace=True,ascending=False )
    fig = px.bar(missing_value_df, x='column_name', y='percent_missing', title="Missing value %",height=700)
    
    fig3.update_layout(title_font_size=30, font_size=15, title_font_family="Arial", legend_title_text='XYZ', margin=dict(b=130))
    
    fig.show()

---

### General Commands

    path = r'xyz.xlsx'
    
    with pd.ExcelWriter(path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name="Sheet_4", index= False)
    
    [df.loc[df['tag'] == i, 'links'] for i in all_links]
    
    [df.loc[df["tag"]==i].links.values[0] for i in l]
    
    df.drop(df[df['Age'] < 25].index, inplace = True)
    
    df.rename(columns=lambda x: x.replace('$', ''), inplace=True)
    
    df.loc[df['id'] == 20 ], 'budget'] = 50
    
    df.loc[df['date_col'].isnull() == True , 'date_col'] = '01/01/1998'
    
    df1['col'].mask(df1['col'] == 'xyz', '', inplace=True)
   
    df.select_dtypes(include=['object']).columns.tolist()
   
    df.select_dtypes(include=np.number).columns.tolist()
    
    with open('dict.pkl', 'wb') as f:
        pickle.dump(d, f)
     
    with open('dict.pkl', 'rb') as f:
        d = pickle.load(f)
        
    -----------------------------------------------------------------------------------------------------------------
        
    df1['code'] = df1['col'].map(d)
    
    df1['col'] = df1['col'].apply(lambda x: '(' + x + ')' if x != " " else "" )
    
    df1['col'] = df1[['col','col1']].apply(lambda x: ''.join(map(str,x)), axis=1)
    
    df1["col"] = ["Yes" if len(df1.col[i]) == 0 else "NO" for i in range(len(df1))]
    
    df1['col'] = df1[['col','col1']].apply(lambda x: ''.join(map(str,x[x.notnull()])), axis=1)
    
    df1['new_col'] = df1['rem_ops'].apply(lambda x: "yes" if 'A' in str(x) else "" ) 
    
    ignore_name =['col1','col2','col3']
    df['special_column'] = df['Special'].apply(lambda x: ','.join(list(set(x.split(",")) - set(ignore_name))))
   
    from collections import Counter
    Counter(train_copy.dtypes.values)
   
    index_names = df[ (df.Device == "xyz") | (df.Device == "xyz")].index
    df.drop(index_names, inplace = True)
    df.reset_index(inplace=True)
    
    df1 = df[(df["Production_line"] == pl_id )  & (df["Task"] == "Operation Change")]
    df1 = df[(df["Production_line"].notnull() )  & (df["Task"] == "Operation Change")]
    
    df["Detail"] = df["Order_Op"].apply(lambda x: x.split("_")[0] if str(x)!="None" else "")  -> 256_AA1 to 256
    
    df1['col'] = df['col'].apply(lambda x: '(' + x + ')' if x != " " else "" )
    df1['col'] = df[['col','col1']].apply(lambda x: ''.join(map(str,x)), axis=1)
    df1['prev_op'] = df1[['prev_op_type','prev_op_end_time']].apply(lambda x: ' '.join(map(str,x[x.notnull()])), axis=1)
    df1['col'] = df1[['col','col1']].apply(lambda x: '/'.join(map(str,x[x.notnull()])), axis=1)
    
    df1['col'] = df1['col'].apply(lambda x: x[:2] if 'xr' in str(x) else x[:1] if str(x)!="nan" else "")
    df1['cc']= df1['machine_id'].apply(lambda x: int(str(x.replace('F',""))))
    
    df1=df_slitter.astype({"CoreCode":str})
    
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
            
    b1 = data.groupby(by=['col1', 'col2']).agg({'Total_Cost': 'sum'}).reset_index()
    b1.columns = [i for i in b1.columns]
     
    list(df3.groupby('id').agg({'product_type':lambda x: list(x)})['product_type'].values)
            
    df = df[~df.machine_name.str.contains("F")]
    
    df[~df.col.isnull()]
    
    for _ , row in df.iterrows():
        print(row.col1, row.col2)
        
    df.reset_index(inplace=True)
    df.groupby('col1').agg({'index':[ 'min', 'max']}).reset_index()
    df_grp.transpose().reset_index(level=0, drop=True).transpose()
    df_grp.rename(columns={ df_grp.columns[0]: "col1" })
   
   
    
    
    
