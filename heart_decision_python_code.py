#!/usr/bin/env python
# coding: utf-8

# # Load the data into Python
# 2. Select the features and the target.
# 3. Create a countplot of the target
# 4. Create one graph for each of the features.
# 5. Create a DecisionTreeClassifier model
# 6. Fit your model
# 7. Predict with your model
# 8. Create a confusion matrix using pd.crosstab
# 9. Print a classification report of the data
# 10. Graph the decision tree

# In[31]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler # Scale: replaces the values by their Z scores
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz 
from IPython.display import Image  
import pydotplus
from sklearn import tree


# In[33]:


df = pd.read_csv("C:\\Users\\titid\\Desktop\\heart_decision.csv")

df.head(10)


# In[34]:


df.describe()


# In[35]:


ax = sns.countplot("target", data = df)
ax


# In[36]:


plt.hist(df.chol)
plt.show()


# In[37]:


plt.hist(df.sex)
plt.show


# In[38]:


plt.hist(df.restecg)
plt.show


# In[39]:


features = df[['sex','chol','restecg']]
target = df.target
features.corr()


# In[41]:


target = df.target
features = df.drop(columns=['target'])

features.head()
target.head
features.head(10)


# # Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.

# In[42]:


sc = preprocessing.StandardScaler()
scaledfeatures = sc.fit_transform(features) # scale our features
features_train, features_test,target_train,target_test = train_test_split(scaledfeatures,target)


# In[43]:


# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(scaledfeatures, target, test_size=0.3, random_state=1) # 70% training and 30% test


# In[44]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)
y_pred


# In[48]:


# Decision tree
#fitting the model
#from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=0)
model.fit(features_train,target_train)
model


# In[51]:


DecisionTreeClassifier(random_state=0)
# predict with your model
predictions = model.predict(features_test)
predictions


# In[50]:


# confusion matrix
predictions = model.predict(features_test)
predictions.shape
pd.crosstab(target_test, predictions,rownames = ['Actual'],colnames = ['Predicted'])


# In[52]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# The F-score, also called the F1-score,
# is a measure of a model's accuracy on a dataset. ...
# The F-score is a way of combining the precision and recall of the model,
# and it is defined as the harmonic mean of the model's precision and recall.

# In[53]:


# classification report
print('\n ** Classification Report * \n\n')
print(classification_report(target_test, predictions))


# In[44]:




dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True, 
                                feature_names=features.columns,  
                                class_names='target')
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png('heart_disease.png')
Image(graph.create_png())


# In[ ]:





# In[ ]:




