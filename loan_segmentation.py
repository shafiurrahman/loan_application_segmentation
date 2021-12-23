import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('train.xlsx')
df.head()
df.info()
df.shape
df.dtypes
df.corr()
 

#Setting Display options to ensure feature name visibility
pd.set_option('display.max_columns',None)

#Warning Suppression
import warnings
warnings.filterwarnings('ignore')

#drop ID
df=df.drop(['ID'],axis=1)

#Defining Target and Independent Features
Y=df[['Approved']]
X=df.drop(['Approved'],axis=1)

#Get the Event Rate
Y.mean()

X['DOB'].max()
X['Lead_Creation_Date'].max()

#process date features
import datetime
reference_date_dob = df.DOB.max() + datetime.timedelta(days = 6458)
print('Reference Date for Age:', reference_date_dob)

reference_date_lead = df.Lead_Creation_Date.max() + datetime.timedelta(days = 30)
print('Reference Date for Lead Creation:', reference_date_lead)


X['age'] = (reference_date_dob - X.DOB).astype('timedelta64[Y]')
X['lead_age']=(reference_date_lead - X.Lead_Creation_Date).astype('timedelta64[M]')
X=X.drop(['DOB','Lead_Creation_Date'],axis=1)
X.head()

X.dtypes

#Split features into Numerical and Categorical
num=X.select_dtypes(include="number")
char=X.select_dtypes(include="object")

num.dtypes
char.dtypes

def unique_levels(x):
    x=x.value_counts().count()
    return(x)
df_value_counts=pd.DataFrame(num.apply(lambda x : unique_levels(x)))
df_value_counts

df_value_counts.columns=['feature_levels']
df_value_counts

slice1=df_value_counts.loc[df_value_counts['feature_levels']<=20]
cat_list=slice1.index
cat=num.loc[:,cat_list]
cat.dtypes

char=pd.concat([char,cat],axis=1,join="inner")
char.head()

num.head()
num.describe(percentiles=[0.01,0.05,0.10,0.25,0.50,0.75,0.85,0.9,0.99])

#Capping and Flooring of outliers
def outlier_cap(x):
    x=x.clip(lower=x.quantile(0.01))
    x=x.clip(upper=x.quantile(0.99))
    return(x)

num=num.apply(lambda x : outlier_cap(x))
num.describe(percentiles=[0.01,0.05,0.10,0.25,0.50,0.75,0.85,0.9,0.99])

#Missing Value Analysis
num.isnull().mean()
num = num.loc[:,num.isnull().mean() <= .25]
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
num_1=pd.DataFrame(imputer.fit_transform(num),index=num.index,columns=num.columns)

num_1.isnull().mean()
char.isnull().mean()
char = char.loc[:,char.isnull().mean() <= .25]

def unique_levels(x):
    x=x.value_counts().count()
    return(x)
char_value_counts=pd.DataFrame(char.apply(lambda x : unique_levels(x)))
char_value_counts

char=char.drop(['City_Code','Employer_Code'],axis=1)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
char_1=pd.DataFrame(imputer.fit_transform(char),index=char.index,columns=char.columns)

char_1.isnull().mean()

#Feature Selection - Numerical Features
#Part 1 : Remove Features with 0 Variance
from sklearn.feature_selection import VarianceThreshold

varselector= VarianceThreshold(threshold=0)
varselector.fit_transform(num_1)
# Get columns to keep and create new dataframe with those only
cols = varselector.get_support(indices=True)
num_1 = num_1.iloc[:,cols]

num_1.iloc[0]
num_1.isnull().sum()

#Part 2 - Bi Variate Analysis (Feature Discretization)
from sklearn.preprocessing import KBinsDiscretizer
discrete=KBinsDiscretizer(n_bins=10,encode='ordinal', strategy='quantile')
num_binned=pd.DataFrame(discrete.fit_transform(num_1),index=num_1.index, columns=num_1.columns).add_suffix('_Rank')
num_binned.head()

#Check if the features show a slope at all
#If they do, then do you see some deciles below the population average and some higher than population average?
#If that is the case then the slope will be strong
#Conclusion: A strong slope is indicative of the features' ability to discriminate the event from non event
#            making it a good predictor

X_bin_combined=pd.concat([Y,num_binned],axis=1,join='inner')

from numpy import mean
for col in (num_binned.columns):
    plt.figure()
    sns.lineplot(x=col,y=X_bin_combined['Approved'].mean(),data=X_bin_combined)
    sns.barplot(x=col, y="Approved",data=X_bin_combined, estimator=mean )
plt.show()

# All features from num_1 will get selected due to good discrimination power by all of them
select_features_df_num=num_1
select_features_df_num.shape

#Feature Selection - Categorical Features
#Part 1 - Bi Variate Analysis
import matplotlib.pyplot as plt
import seaborn as sns
X_char_merged=pd.concat([Y,char_1],axis=1,join='inner')

from numpy import mean
for col in (char.columns):
    plt.figure()
    sns.lineplot(x=col,y=X_char_merged['Approved'].mean(),data=X_char_merged)
    sns.barplot(x=col, y="Approved",data=X_char_merged, estimator=mean )
plt.show()

char_1=char_1.drop(['Source','Customer_Existing_Primary_Bank_Code'],axis=1)
# Create dummy features with n-1 levels
X_char_dum = pd.get_dummies(char_1, drop_first = True)
X_char_dum.shape

#Part 2 - Select K Best
# Select K Best for Categorical Features
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=23)
selector.fit_transform(X_char_dum, Y)
# Get columns to keep and create new dataframe with those only
cols = selector.get_support(indices=True)
select_features_df_char = X_char_dum.iloc[:,cols]

select_features_df_char=X_char_dum

#Creating the Master Feature Set for Model Development
X_all=pd.concat([select_features_df_char,select_features_df_num],axis=1,join="inner")
X_all.shape
X_all.head()

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_all, Y, test_size=0.3, random_state=20)

print("Shape of Training Data",X_train.shape)
print("Shape of Testing Data",X_test.shape)
print("Response Rate in Training Data",y_train.mean())
print("Response Rate in Testing Data",y_test.mean())



#MODEL BUILDING
import statsmodels.api as sm
log_reg = sm.Logit(y_train, X_train).fit()
print(log_reg.summary())

#Warning: Did you receive an error of the form "LinAlgError: Singular matrix"? 
# This means that statsmodels was unable to fit the model due to certain linear algebra computational problems. 
# Specifically, the matrix was not invertible due to not being full rank. In other words, there was a lot of redundant, superfluous data. Try removing some features from the model and running it again.,
# Create a new model, this time only using those features you determined were influential based on your analysis of the results above. How does this model perform?,Only consider the columns specified in relevant_columns when building your model. The next step is to create dummy variables from categorical variables. Remember to drop the first level for each categorical column and make sure all the values are of type float: ,Now with everything in place, you can build a logistic regression model using statsmodels (make sure you create an intercept term as we showed in the previous lesson).

# Building a Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(criterion='gini',random_state=0)

dtree=DecisionTreeClassifier(criterion='gini',random_state=0,max_depth=4,min_samples_split=500)
dtree.fit(X_train,y_train)

#!pip install pydotplus

from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
plt.figure(figsize=[50,10])
tree.plot_tree(dtree,filled=True,fontsize=15,rounded=True,feature_names=X_all.columns)
plt.show()

# Model Evaluation
y_pred_tree=dtree.predict(X_test)

from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_tree))
print("Precision",metrics.precision_score(y_test,y_pred_tree))
print("Recall",metrics.recall_score(y_test,y_pred_tree))
print("f1_score",metrics.f1_score(y_test,y_pred_tree))

metrics.plot_confusion_matrix(dtree,X_all,Y)

# Gains Chart Analysis
# This will help understand the power of discrimination offered by the model's estimated probabilities

# Logistic Regression - Gains Chart
y_pred_prob = dtree.predict_proba(X_all)[:, 1]
df['pred_prob_dtree']=pd.DataFrame(y_pred_prob)
df['P_Rank_tree']=pd.qcut(df['pred_prob_dtree'].rank(method='first').values,20,duplicates='drop').codes+1
rank_df_actuals=df.groupby('P_Rank_tree')['Approved'].agg(['count','mean'])
rank_df_predicted=df.groupby('P_Rank_tree')['pred_prob_dtree'].agg(['mean'])
rank_df_actuals=pd.DataFrame(rank_df_actuals)

rank_df_actuals.rename(columns={'mean':'Actual_event_rate'},inplace=True)
rank_df_predicted=pd.DataFrame(rank_df_predicted)

rank_df_predicted.rename(columns={'mean':'Predicted_event_rate'},inplace=True)
rank_df=pd.concat([rank_df_actuals,rank_df_predicted],axis=1,join="inner")

sorted_rank_df=rank_df.sort_values(by='P_Rank_tree',ascending=False)
sorted_rank_df['N_events']=rank_df['count']*rank_df['Actual_event_rate']
sorted_rank_df['cum_events']=sorted_rank_df['N_events'].cumsum()
sorted_rank_df['event_cap']=sorted_rank_df['N_events']/max(sorted_rank_df['N_events'].cumsum())
sorted_rank_df['cum_event_cap']=sorted_rank_df['event_cap'].cumsum()

sorted_rank_df['N_non_events']=sorted_rank_df['count']-sorted_rank_df['N_events']
sorted_rank_df['cum_non_events']=sorted_rank_df['N_non_events'].cumsum()
sorted_rank_df['non_event_cap']=sorted_rank_df['N_non_events']/max(sorted_rank_df['N_non_events'].cumsum())
sorted_rank_df['cum_non_event_cap']=sorted_rank_df['non_event_cap'].cumsum()

sorted_rank_df['KS']=round((sorted_rank_df['cum_event_cap']-sorted_rank_df['cum_non_event_cap']),4)

sorted_rank_df['random_cap']=sorted_rank_df['count']/max(sorted_rank_df['count'].cumsum())
sorted_rank_df['cum_random_cap']=sorted_rank_df['random_cap'].cumsum()
sorted_reindexed=sorted_rank_df.reset_index()
sorted_reindexed['Decile']=sorted_reindexed.index+1
sorted_reindexed

ax = sns.lineplot( x="Decile", y="cum_random_cap", data=sorted_reindexed,color='red')
ax = sns.lineplot( x="Decile", y="cum_event_cap", data=sorted_reindexed,color='grey')

df['Predicted_Approval_Rank']=np.where(df['P_Rank_tree']>=18,"Top 3","Bottom 7")
df['Predicted_Approval_Rank_2']=np.where(df['P_Rank_tree']>=18,"Top 3",np.where(df['P_Rank_tree']>=16,"Mid 2","Bottom Rest"))
df.groupby('Predicted_Approval_Rank_2')['Approved'].agg(['mean','count'])
df.groupby('P_Rank_tree')['pred_prob_dtree'].agg(['min','max','mean'])

# Concluding Notes
# The business team can notify the Loan Approval team to essentially target the applications which fall under 'Top 3' 
# followed by Mid 2 deciles

# Probability Cutoff to be applied
# Phase 1 - Focus on Prob value >=0.031460 [ Near about 10457 loan applications]
# Phase 2 - Focus on Prob value >=0.016718 and <=0.031460 [ Near about 6971 Loan applications ]

df.head()
chk=df.loc[df['pred_prob_dtree']>=0.031460,:]
chk.shape