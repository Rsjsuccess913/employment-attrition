#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[4]:


Emp_data=pd.read_csv(r"C:\Users\lenovo\Downloads\Dataset01-Employee_Attrition.csv")
Emp_data.head()


# In[5]:


Emp_data.shape


# In[6]:


Emp_data.columns


# In[7]:


Emp_data.dtypes


# In[8]:


Emp_data.info()


# In[9]:


Emp_data[Emp_data.duplicated()]


# In[10]:


Emp_data1=Emp_data.drop_duplicates()
Emp_data1.shape


# In[11]:


Emp_data1.isnull().sum()


# In[13]:


Emp_data1['left'].value_counts()


# In[14]:


Emp_data1['left'].value_counts().plot(kind='bar')


# In[15]:


Emp_data1.head()


# In[16]:


pd.crosstab(Emp_data1.salary,Emp_data1.left).plot(kind='bar')


# In[17]:


pd.crosstab(Emp_data1.salary,Emp_data1.left)


# In[18]:


pd.crosstab(Emp_data1.salary,Emp_data1.left).plot(kind='bar')


# In[21]:


pd.crosstab(Emp_data1.Department,Emp_data1.left).plot(kind='bar')


# In[22]:


pd.crosstab(Emp_data1.Department,Emp_data1.left)


# In[23]:


num_feature_list1=[f for f in Emp_data1.columns if Emp_data1.dtypes[f]=='float64']
num_feature_list1


# In[24]:


num_feature_list2=[f for f in Emp_data1.columns if Emp_data1.dtypes[f]=='int64']
num_feature_list2


# In[37]:


num_col_list=['number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','satisfaction_level','last_evaluation']


# In[38]:


fig,axes=plt.subplots(ncols=4,figsize=(12,3))
for column,axis in zip(num_col_list[:4],axes):
    sns.boxplot(data=Emp_data1[column],ax=axis)
    axis.set_title(column)
plt.tight_layout()
plt.show()


# In[39]:


ig,axes=plt.subplots(ncols=3,figsize=(12,3))
for column,axis in zip(num_col_list[4:],axes):
    sns.boxplot(data=Emp_data1[column],ax=axis)
    axis.set_title(column)
plt.tight_layout()
plt.show()


# In[40]:


Emp_data1['number_project'].plot(kind='hist',bins=5)


# In[41]:


Emp_data1['average_montly_hours'].plot(kind='hist',bins=6)


# In[42]:


Emp_data1['time_spend_company'].plot(kind='hist',bins=5)


# In[43]:


Emp_data1['satisfaction_level'].plot(kind='hist',bins=5)


# In[44]:


Emp_data1['last_evaluation'].plot(kind='hist',bins=5)


# In[45]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()


# In[47]:


Emp_data1['salary']=label_encoder.fit_transform(Emp_data1['salary'])
Emp_data1['department']=label_encoder.fit_transform(Emp_data1['Department'])


# In[48]:


Emp_data1.head()


# In[59]:


x=Emp_data1.drop('left',axis=1)
y=Emp_data1['left']


# In[60]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[61]:


x_train.shape


# In[62]:


x_train.head()


# In[63]:


from sklearn.preprocessing import StandardScaler
std_scaler=StandardScaler()


# In[69]:


x_train_encoded = pd.get_dummies(x_train)
x_test_encoded = pd.get_dummies(x_test)

x_train_encoded, x_test_encoded = x_train_encoded.align(x_test_encoded, join='left', axis=1, fill_value=0)

xtrain_scaled = std_scaler.fit_transform(x_train_encoded)
xtest_scaled = std_scaler.transform(x_test_encoded)


# In[70]:


xtrain_scaled
xtest_scaled


# In[72]:


from sklearn.ensemble import RandomForestClassifier


# In[73]:


Random_forest_model=RandomForestClassifier()


# In[74]:


Random_forest_model.fit(xtrain_scaled,y_train)


# In[77]:


y_pred=Random_forest_model.predict(xtest_scaled)
y_pred


# In[79]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[80]:


sns.heatmap(cm,annot=True,fmt='d')


# In[82]:


from sklearn.metrics import accuracy_score
model_accuracy=accuracy_score(y_test,y_pred)
print("Accuracy of the model=",model_accuracy)


# In[84]:


from sklearn.metrics import precision_score
model_precision=precision_score(y_test,y_pred)
print("precision of the model=",model_precision)


# In[86]:


from sklearn.metrics import recall_score
model_recall=recall_score(y_test,y_pred)
print("recall of the model=",model_recall)


# In[87]:


from sklearn.metrics import f1_score
f1score=f1_score(y_test,y_pred)
print("f1 score of the model=",f1score)


# In[88]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[139]:


list_of_features = x_train_encoded.columns  # use same data as used in model.fit()

score_list = Random_forest_model.feature_importances_

print("Number of features:", len(list_of_features))
print("Number of importance scores:", len(score_list))

import pandas as pd

score_df = pd.DataFrame({
    "Feature": list_of_features,
    "Score": score_list
})

score_df = score_df.sort_values(by='Score', ascending=False)

print(score_df)


# In[105]:


list_of_figures=list(x.columns)
plt.figure(figsize=(10,16))
plt.barh(range(len(list_of_features)),list_of_features)
plt.ylabel('Features')
plt.show()


# In[106]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(Random_forest_model,xtrain_scaled,y_train,cv=5,scoring='accuracy')
print('cross-validation scores=',scores)


# In[108]:


Avg_model_score=scores.mean()
print("Average Modal Score=",Avg_model_score)


# In[130]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf_model = RandomForestClassifier()


param_grid = {
    'n_estimators': [50, 100, 200],  
    'max_depth': [5, 10, None]
}


grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)


grid_search.fit(xtrain_scaled, y_train)


# In[131]:


grid_search.best_params_


# In[129]:





# In[132]:


Random_forest_model_new=RandomForestClassifier(max_features='sqrt',n_estimators=50)


# In[133]:


Random_forest_model_new.fit(xtrain_scaled,y_train)


# In[135]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(Random_forest_model_new,xtrain_scaled,y_train,cv=5,scoring='accuracy')
print('cross-validation scores=',scores)


# In[136]:


Avg_model_score=scores.mean()


# In[138]:


print('Average Model Score=',Avg_model_score)


# In[ ]:




