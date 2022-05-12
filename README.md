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
    
    
    
