import numpy as np
import pandas as pd

#import datasets
df = pd.read_csv('Train_set for big_mart.csv')
df1 = pd.read_csv('Test_set for big_mart.csv')

#combining train and test set so dont have to repeat same step twice for train and test set
df['source']='df'
df1['source']='df1'
data=pd.concat([df,df1],ignore_index=True)
print(df.shape,df1.shape, data.shape)

data.apply(lambda x: sum(x.isnull()))
#missing values in Item_Weight, Outlet_Size

#checking unique values in each of them
data.apply(lambda x:len(x.unique()))

data.dtypes
categorical_columns=[x for x in data.dtypes.index if data.dtypes[x]=='object']
#excluding mixdatatypes columns and source column
categorical_columns=[x for x in categorical_columns if x not in['Item_Identifier','source','Outlet_Identifier']]
#print value count of each of categorical_columns
for col in categorical_columns:
    print("Frequencies of categories for variable %s" %col )
    print(data[col].value_counts())
#item fat content: some low fat is written as LF and low fat and regualr as reg
    
#now take care for missing values and outliers
#only these two have missing values ,leave item outlet sales as it is the output variable
data['Item_Weight'].fillna((data['Item_Weight'].mean()),inplace=True)
data['Outlet_Size'].fillna('Medium',inplace=True)

#item visibility 0 makes no sense, take care of this
# consider 0 like missing values and fill it with mean
data['Item_Visibility'].value_counts()  #879 o's are there
data.Item_Visibility = data.Item_Visibility.replace({0.000000: data['Item_Visibility'].mean()})

#create new category by item type
#item identifier has three categories FD(food),dr(drinks) and NC(nonconsumable)
#firstly get the first two characters if ID:
data['Item_Type_Combined']=data['Item_Identifier'].apply(lambda x:x[0:2])
#now rename them to more understandable cayegories
data['Item_Type_Combined']=data['Item_Type_Combined'].map({'FD':'Food','NC':'Non consumables','DR':'Drinks'})
data['Item_Type_Combined'].value_counts() 

#now determine the years of operation of the store
data['Outlet_Establishment_Year'].value_counts() 
data['Outlet_Year']=2013-data['Outlet_Establishment_Year'] #its 2013's data
data['Outlet_Year'].describe()

#now modify item fat content categories
data['Item_Fat_Content'].value_counts()
data['Item_Fat_Content']=data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})

#there are some non consumable item,so no fat content
data.loc[data['Item_Type_Combined'] == "Non consumables",'Item_Fat_Content'] = "Non Edibles"
data['Item_Fat_Content'].value_counts()

#encoding categorical data
#label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Outlet']=le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Item_Type_Combined','Outlet_Size','Outlet_Location_Type','Outlet_Type']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])
data.dtypes 

#now one hot encoding to get dummy variables
data=pd.get_dummies(data,columns=['Item_Fat_Content','Item_Type_Combined','Outlet_Size','Outlet_Location_Type','Outlet_Type','Outlet'])
data.dtypes

#drop columns which has been converted into different types
data.drop(['Item_Type','Outlet_Establishment_Year'], axis=1 ,inplace=True)

#now convert data back into train and test set
X_train=data.loc[data['source']=="df"]
X_test=data.loc[data['source']=="df1"]
y_train=X_train.iloc[:,2:3]

#now drop output column and source from train and test set
X_train.drop(['Item_Outlet_Sales','source','Item_Identifier'],axis=1,inplace=True)
X_test.drop(['Item_Outlet_Sales','source','Item_Identifier'],axis=1,inplace=True)

X_train.drop(['Outlet_Identifier'],axis=1,inplace=True)
X_test.drop(['Outlet_Identifier'],axis=1,inplace=True)

#fitiing random forest regression to the training set
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=400,max_depth=6,min_samples_leaf=100,n_jobs=4)
regressor.fit(X_train,y_train)

#compute y_test
y_test=regressor.predict(X_test)
y_test=pd.Series(y_test)

#to convert series into csv 
sample=pd.Series.to_csv(y_test,index=None)

