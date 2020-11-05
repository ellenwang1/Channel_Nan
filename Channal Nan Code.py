#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[2]:


#base libraries
import os, json
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from pandas_profiling import profile_report
from dateutil.parser import parse

#visualisation libraries
import matplotlib as plt
import seaborn as sns

print("imported")


# # Merging data frames

# In[4]:


#Creating the data frame
pd.set_option('display.max_columns', None)

#Read files and create data frame
data = pd.DataFrame()
file_list = glob.glob('Dataset/Dataset/Data/*')

dfs = []
for file in file_list:
    data = pd.read_csv(file, compression='gzip', header=0, sep=',', quotechar='"')
    dfs.append(data)

#Concatenate all data frames as one
data = pd.concat(dfs, ignore_index=True) 

print("done")


# # Brief understanding of the dataset

# In[38]:


#Generate report for entire data frame
data.profile_report()


# In[5]:


#Create dataframe for obvious vs. non-obvious bots using self-declaration 
data_bot = data.query('br_name == "Robot/Spider"')
non_data_bot = data.query('br_name != "Robot/Spider"')

#Create column for binary is_bot and is_not bot
data['is_bot'] = np.where(data["br_name"] == "Robot/Spider", 0, 1)
print("created")


# In[84]:


#Generate report for data_bot data frame
data_bot.profile_report()


# In[85]:


#Generate report for non_data_bot data frame
non_data_bot.profile_report()


# In[39]:


#Using describe function to look at numerical columns in dataframe
data.describe()


# # Exploratory Data Analysis on all variables

# In[6]:


#Looking at member_id counts - unique values? for bots
data_bot.member_id.value_counts()
non_data_bot.member_id.value_counts()


# In[ ]:


#Looking at member_id counts for non-bots
non_data_bot.member_id.value_counts()


# ######  Observations from member_id #####
# 1. Highest count of member_id are actually NULL values but hashed using a key
# 2. Not a useful predictor because these values are ultimately still unique
# 3. The duplicates come from multiple visits per account

# In[67]:


#Looking at domain_userid counts for bots
data_bot.domain_userid.value_counts()


# In[68]:


#Looking at domain_userid counts for bots
non_data_bot.domain_userid.value_counts()


# ######  Observations from domain_userid #####
# 1. Majority of data bots have the domain_userid "84896d88-79b6-4a07-92a4-82a6352fa98d"
# 2. This is likely a good variable to use in our regression analysis

# In[87]:


#Closer look at derived_tstamp and collector t_stamp
print(data.derived_tstamp)


# In[11]:


#Changing data format of tstamp from UTC to datetime-timestamp in the data_bot dataframe
data_bot['collector_tstamp'] = data_bot['collector_tstamp'].apply(lambda x: parse(x))
data_bot['derived_tstamp'] = data_bot['derived_tstamp'].apply(lambda x: parse(x))
print("complete")


# In[12]:


#Changing data format of tstamp from UTC to datetime-timestamp in the non_data_bot dataframe
non_data_bot['collector_tstamp'] = non_data_bot['collector_tstamp'].apply(lambda x: parse(x))
non_data_bot['derived_tstamp'] = non_data_bot['derived_tstamp'].apply(lambda x: parse(x))
print("complete")


# In[15]:


#Timestamp bins in 5 minute intervals for all data variables for t_stamp values
count_data_bot_all = data_bot.resample('5T', on='collector_tstamp').count()
count_data_bot_all


# In[16]:


#Graph of all variables and their frequency against time based on 5 minute bins
count_data_bot_all.plot()


# In[ ]:


#Timestamp bins in Months intervals for all data variables for t_stamp values for data bots
#Counting frequency of visits within each timestamp bin
count_data_bot_dstamp = data_bot.resample('M', on='derived_tstamp').count()['domain_userid']
count_data_bot_dstamp


# In[166]:


count_data_bot_dstamp.plot()


# In[156]:


#Timestamp bins in 5 minute time intervals for all data variables for d_stamp values for non data bots
#Counting frequency of visits within each timestamp bin
count_non_data_bot_dstamp = non_data_bot.resample('M', on='derived_tstamp').count()['domain_userid']
count_non_data_bot_dstamp


# In[167]:


count_non_data_bot_dstamp.plot()


# In[142]:


#Timestamp bins in 5 minute time intervals for all data variables for t_stamp values for data bots
#Counting frequency of visits within each timestamp bin
count_data_bot_tstamp = data_bot.resample('5T', on='collector_tstamp').count()['domain_userid']
count_data_bot_tstamp


# In[143]:


count_data_bot_tstamp.plot()


# In[48]:


#Timestamp bins in 5 minute time intervals for all data variables for t_stamp values for non data bots
#Counting frequency of visits within each timestamp bin
count_non_data_bot_tstamp = non_data_bot.resample('5T', on='collector_tstamp').count()['domain_userid']
count_non_data_bot_tstamp


# In[138]:


count_non_data_bot_tstamp.plot()


# In[54]:


#dataset analysis
data_bot.network_userid.value_counts()


# In[53]:


non_data_bot.network_userid.value_counts()


# In[55]:


data_bot.domain_sessionid.value_counts()


# In[57]:


non_data_bot.domain_sessionid.value_counts()


# In[62]:


data_bot.useragent.value_counts()


# In[63]:


non_data_bot.useragent.value_counts()


# ######  Observations from useragent #####
# 1. Each entry self declares through "spider", "crawler", "bot"

# In[69]:


data_bot.os_name.value_counts()


# In[70]:


non_data_bot.os_name.value_counts()


# In[71]:


data_bot.os_family.value_counts()


# In[72]:


non_data_bot.os_name.value_counts()


# In[73]:


data_bot.br_type.value_counts()


# In[74]:


non_data_bot.os_family.value_counts()


# In[90]:


#Suspicions on visitors who are classified as non-bots
unknown_data_bot = non_data_bot.query('os_family == "Unknown" and domain_userid == "84896d88-79b6-4a07-92a4-82a6352fa98d"')
unknown_data_bot


# In[91]:


#Number of suspicious visitors in non_data_bot dataframe 43/1441 number defined under domain_userid, might not be significant
len(unknown_data_bot)


# In[92]:


data_bot.br_type.value_counts()


# In[93]:


non_data_bot.br_type.value_counts()


# In[101]:


data_bot.page_urlpath.value_counts()


# In[102]:


non_data_bot.page_urlpath.value_counts()


# ######  Observations from page_urlpath #####
# 1. Databots visit shorter page_urlpaths (typically < 30 character length)

# In[119]:


data_bot.user_ipaddress.value_counts()


# In[120]:


non_data_bot.user_ipaddress.value_counts()


# In[121]:


data_bot.page_view_id.value_counts()


# In[122]:


non_data_bot.page_view_id.value_counts()


# In[123]:


data_bot.member_id.value_counts()


# In[125]:


non_data_bot.member_id.value_counts()


# In[126]:


data_bot.member_type.value_counts()


# In[127]:


non_data_bot.member_type.value_counts()


# In[131]:


data_bot.article_type.value_counts()


# In[130]:


non_data_bot.article_type.value_counts()


# ######  Observations from article_type #####
# 1. Databots tend to visit INDEX page more often than non-databots

# In[132]:


#Looking at the number of non_data_bots who accessed the index page
index_people = non_data_bot.query('article_type == "INDEX"')
index_people


# In[133]:


#Generating a profile_report using pandas_profiling for the new dataframe
index_people.profile_report()


# In[135]:


data_bot.article_primary_category.value_counts()


# In[136]:


non_data_bot.article_primary_category.value_counts()


# # Preparing data for logistic regression

# In[154]:


get_ipython().system('pip install scikit-learn')
print("sklearn installed")


# In[19]:


from sklearn.model_selection import train_test_split

#Splitting the data randomly into a testing and training set
train_df, test_df = train_test_split(data, test_size=0.2)
#Using a copy of the train_df and test_df
train_data = train_df.copy()
test_df = test_df.copy()


# In[182]:


len(train_data)


# In[183]:


len(test_data)


# In[20]:


#missng values analysis in train_data
train_data.isnull().sum()


# In[21]:


#Dropping obviously irrelevant columns from the dataframe
train_data.drop(columns=["page_urlscheme", "page_urlhost", "page_urlport",
                         "refr_urlscheme", "refr_urlhost", "refr_urlport", 
                         "refr_urlpath", "network_userid", "br_name", 
                         "br_family", "br_type", "article_id", 
                         "article_primary_category", "app_id", "member_id"], axis=1, inplace=True)
train_data.drop("domain_sessionid", axis=1, inplace=True)
train_data.drop("user_ipaddress", axis=1, inplace=True)
train_data.drop("useragent", axis=1, inplace=True)


# In[22]:


train_data.isnull().sum()


# In[23]:


#Pad NULL values in br_version column with -1 values
train_data["br_version"] = train_data["br_version"].fillna(-1)

#Pad NULL values in page_view_id column with 0
train_data["page_view_id"] = train_data["page_view_id"].fillna(0)

#Pad NULL values in article_type column with "Error"
train_data["article_type"] = train_data["article_type"].fillna("ERROR")


# In[24]:


train_data.isnull().sum()


# In[191]:


#Looking at first 5 observations
train_data.head()


# In[25]:


#Creating dummy variables for categorical groups

#domain_userid
train_data['domain_userid'] = np.where(train_data['domain_userid'] == "84896d88-79b6-4a07-92a4-82a6352fa98d", 1, 0)
#os_name
train_data['os_name'] = np.where(train_data['os_name'] == "unknown", 1, 0)
#os_family
train_data['os_family'] = np.where(train_data['os_family'] == "unknown", 1, 0)
#br_version
train_data['br_version'] = np.where(train_data['br_version'] == -1, 1, 0)
#page_urlpath
train_data['path_len'] = train_data['page_urlpath'].apply(lambda x: len(x))
train_data['path_len_article'] = np.where(train_data['path_len'] > 40, 1, 0)
#member_type
train_data['member_type'] = np.where(train_data['member_type']=="Subscriber", 1, 0)
#article_type
train_data['article_type_index'] = np.where(train_data['article_type']=="INDEX", 1, 0)
train_data['article_type_article'] = np.where(train_data['article_type']=="ARTICLE", 1, 0)
train_data['article_type_homepage'] = np.where(train_data['article_type']=="HOMEPAGE", 1, 0)


# In[194]:


train_data.head()


# In[27]:


#Looking at the difference between d_stamp and t_stamp and using that value as a variable
train_data['collector_tstamp'] = train_data['collector_tstamp'].apply(lambda x: parse(x))
train_data['derived_tstamp'] = train_data['derived_tstamp'].apply(lambda x: parse(x))
train_data['time_diff'] = train_data['collector_tstamp'] - train_data['derived_tstamp']
train_data['time_diff'] = train_data['time_diff'] / np.timedelta64(1, 's')


# In[33]:


train_data


# In[196]:


#Dropping some more variables after dummy variable conversions
train_data.drop(columns=["derived_tstamp", "collector_tstamp", "page_urlpath", "page_view_id", "article_type", "path_len_article"], axis=1, inplace=True)


# In[34]:


train_data.head()


# In[38]:


#Final train dataframe for the model
final_train = train_data


# In[36]:


#Test_data copy all transformations
test_data = test_df.copy()
test_data.drop(columns=["page_urlscheme", "page_urlhost", "page_urlport","refr_urlscheme", "refr_urlhost", "refr_urlport", "refr_urlpath", "network_userid", "br_name", "br_family", "br_type", "article_id", "article_primary_category", "app_id", "member_id"], axis=1, inplace=True)
test_data["br_version"] = test_data["br_version"].fillna(-1)
test_data["page_view_id"] = test_data["page_view_id"].fillna(0)
test_data["article_type"] = test_data["article_type"].fillna("ERROR")
test_data['domain_userid'] = np.where(test_data['domain_userid'] == "84896d88-79b6-4a07-92a4-82a6352fa98d", 1, 0)
test_data.drop("domain_sessionid", axis=1, inplace=True)
test_data.drop("user_ipaddress", axis=1, inplace=True)
test_data.drop("useragent", axis=1, inplace=True)

#os_name
test_data['os_name'] = np.where(test_data['os_name'] == "unknown", 1, 0)
#os_family
test_data['os_family'] = np.where(test_data['os_family'] == "unknown", 1, 0)
#br_version
test_data['br_version'] = np.where(test_data['br_version'] == -1, 1, 0)
#page_urlpath
test_data['path_len'] = test_data['page_urlpath'].apply(lambda x: len(x))
test_data['path_len_article'] = np.where(test_data['path_len'] > 40, 1, 0)
#member_type
test_data['member_type'] = np.where(test_data['member_type']=="Subscriber", 1, 0)
#article_type
test_data['article_type_index'] = np.where(test_data['article_type']=="INDEX", 1, 0)
test_data['article_type_article'] = np.where(test_data['article_type']=="ARTICLE", 1, 0)
test_data['article_type_homepage'] = np.where(test_data['article_type']=="HOMEPAGE", 1, 0)
#time_diff
#Looking at the difference between d_stamp and t_stamp and using that value as a variable
test_data['collector_tstamp'] = test_data['collector_tstamp'].apply(lambda x: parse(x))
test_data['derived_tstamp'] = test_data['derived_tstamp'].apply(lambda x: parse(x))
test_data['time_diff'] = test_data['collector_tstamp'] - test_data['derived_tstamp']
test_data['time_diff'] = test_data['time_diff'] / np.timedelta64(1, 's')

#Dropping some more final variables
test_data.drop(columns=["derived_tstamp", "collector_tstamp", "page_urlpath", "page_view_id", "article_type", "path_len_article"], axis=1, inplace=True)

final_test = test_data


# In[42]:


#Final variables for model
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
cols = ["domain_userid", "os_name", "os_family", 
                     "br_version", "member_type", "path_len", 
                     "article_type_index", "article_type_article", 
                     "article_type_homepage", "time_diff"]


# # Running the Logistic Regression Model

# In[48]:


#Assign all independent variables to X and dependent variable (dummy) is_bot to y
X = final_train[cols]
y = final_train['is_bot']

#Use a Logistic regression
model = LogisticRegression(max_iter=1000, verbose=10)

#Looking at the 10 variables in the model
rfe = RFE(model, 10)

#Fit the Logistic regression model to X and y
rfe = rfe.fit(X, y)

print('Selected features: %s' % list(X.columns[rfe.support_]))


# In[44]:


#Look at the variable rankings by order using rfe.ranking_ 
#Probably keep 5 variables
print(rfe.ranking_)


# In[ ]:


#Looking at the top 5 variables in the model
rfe = RFE(model, 5)

#Fit the Logistic regression model to X and y
rfe = rfe.fit(X, y)

print('Selected features: %s' % list(X.columns[rfe.support_]))


# In[ ]:


#Look at coefficients of these variables
print ('coefficients',rfe.estimator_.coef_)


# # Selecting features using RFECV

# #### Code keeps crashing here: max_iter problem? Not sure 

# In[ ]:


from sklearn.feature_selection import RFECV
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=LogisticRegression(max_iter=150), step=1, cv=10, scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))


# In[ ]:


# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# # Fitting a Logistic Regression Model to all variables (Not optimized)

# In[212]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


logreg.coef_

