#!/usr/bin/env python
# coding: utf-8

# ## The below code utilizing risk event description to derive risk taxonomy categories using supervised learning

# In[215]:


# Data Structures
import pandas as pd
import numpy as np

# Utilities
from pprint import pprint

# Visualizations 
import seaborn as sns
import matplotlib.pylab as plt

# Machine Learing 
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit


# ## Reading in the Risk Events data

# In[3]:


risk_events = pd.read_csv('Risk Events.csv',header=0,encoding = 'unicode_escape')


# In[4]:


risk_events.shape


# In[5]:


risk_events.head()


# In[6]:


risk_events.columns.tolist()


# ## Data Cleaning

# In[120]:


#Select required columns
df1 = risk_events[['Event ID','Risk Taxonomy Level 1','Reporting Description']]
df1.shape


# In[121]:


#rename columns
cols = ['event_id','risk_taxonomy','description']
df1.columns = cols
df1.isnull().sum()


# In[122]:


print(len(df1['risk_taxonomy'].value_counts()))


# In[123]:


df1['risk_taxonomy'].value_counts()


# In[124]:


df1.loc[(df1.risk_taxonomy == '8. Information Security'),'risk_taxonomy']='Information Security'
df1.loc[(df1.risk_taxonomy == '9. Information Technology'),'risk_taxonomy']='Information Technology'
df1.loc[(df1.risk_taxonomy == '15. Model'),'risk_taxonomy']='Model'
df1.loc[(df1.risk_taxonomy == '3. Business Continuation'),'risk_taxonomy']='Business Continuation'
df1.loc[(df1.risk_taxonomy == '12. Legal and Regulatory'),'risk_taxonomy']='Legal and Regulatory'
df1.loc[(df1.risk_taxonomy == '1. Accounting and Financial Reporting'),'risk_taxonomy']='Accounting and Financial Reporting'


# ## Visualizing the distribution of Target Classes

# In[187]:


df2 = df1[df1['risk_taxonomy'].isin(['External Fraud', 'Execution, Delivery and Process Management',
                                     'Clients, Products and Business Practices','Legal and Regulatory',
                                     'Information Technology','7. Fraud',
                                     'Outsourcing, Vendors, Counterparties and Suppliers','16. Operations','Internal Fraud',
                                     'Information Security'])]


# In[194]:


df2_value_counts = df2['risk_taxonomy'].value_counts()


# In[195]:


type(df2_value_counts)


# In[196]:


df2_sub = df2_value_counts.rename_axis('risk_taxonomy').reset_index(name='counts')
df2_sub.columns


# In[200]:


df2_sub


# In[211]:


sns.set_context('paper')
 
plt.xticks(rotation  = 45)

sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(10,10)})

sns.barplot(x="risk_taxonomy", y="counts", data=df2_sub)


# ## Undersample the majority classes

# In[213]:


# Separate majority and minority classes
df_1 = df2[df2.risk_taxonomy== "External Fraud"]
df_2 = df2[df2.risk_taxonomy== "Execution, Delivery and Process Management"]
df_3 = df2[df2.risk_taxonomy== "Clients, Products and Business Practices"]
df_4 = df2[df2.risk_taxonomy== "Legal and Regulatory"]
df_5 = df2[df2.risk_taxonomy== "Information Technology"]
df_6 = df2[df2.risk_taxonomy== "7. Fraud"]
df_7 = df2[df2.risk_taxonomy== "Outsourcing, Vendors, Counterparties and Suppliers"]
df_8 = df2[df2.risk_taxonomy== "16. Operations"]
df_9 = df2[df2.risk_taxonomy== "Information Security"]
df_10 = df2[df2.risk_taxonomy== "Internal Fraud"]

# Downsample majority class
df_1_downsampled = resample(df_1,replace=False,    # sample without replacement
                                 n_samples=130,     # to match minority class
                                 random_state=123) # reproducible results
df_2_downsampled = resample(df_2,replace=False,    # sample without replacement
                                 n_samples=130,     # to match minority class
                                 random_state=123) # reproducible results
df_3_downsampled = resample(df_3,replace=False,    # sample without replacement
                                 n_samples=130,     # to match minority class
                                 random_state=123) # reproducible results
df_4_downsampled = resample(df_4,replace=False,    # sample without replacement
                                 n_samples=130,     # to match minority class
                                 random_state=123) # reproducible results
df_5_downsampled = resample(df_5,replace=False,    # sample without replacement
                                 n_samples=130,     # to match minority class
                                 random_state=123) # reproducible results
df_6_downsampled = resample(df_6,replace=False,    # sample without replacement
                                 n_samples=130,     # to match minority class
                                 random_state=123) # reproducible results
df_7_downsampled = resample(df_7,replace=False,    # sample without replacement
                                 n_samples=130,     # to match minority class
                                 random_state=123) # reproducible results
df_8_downsampled = resample(df_8,replace=False,    # sample without replacement
                                 n_samples=130,     # to match minority class
                                 random_state=123) # reproducible results
df_9_downsampled = resample(df_9,replace=False,    # sample without replacement
                                 n_samples=130,     # to match minority class
                                 random_state=123) # reproducible results


# In[216]:


# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_1_downsampled,
                            df_2_downsampled,
                           df_3_downsampled,
                           df_4_downsampled,
                           df_5_downsampled,
                           df_6_downsampled,
                           df_7_downsampled,
                           df_8_downsampled,
                           df_9_downsampled,df_10])
 
# Display new class counts
df_downsampled.risk_taxonomy.value_counts()


# In[220]:


df_downsampled_sub = df_downsampled.risk_taxonomy.value_counts().rename_axis('risk_taxonomy').reset_index(name='counts')
df_downsampled_sub


# In[224]:


sns.set_context('paper')
 
plt.xticks(rotation  = 45)

sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(5,10)})

sns.barplot(x="risk_taxonomy", y="counts", data=df_downsampled_sub)


# In[226]:


df_downsampled.head(20)


# ## Natural Language Processing

# In[227]:


import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np

from nltk.corpus import stopwords
from gensim.parsing import preprocessing as pp

stop_words = stopwords.words("english")
stop_words = set(stop_words)


# ### 1. Converting text to tokens
# ### 2. Removing 
# ### 3. Stripping short words .. 

# In[228]:


pp_list = [
    lambda x: x.lower(),
    pp.strip_tags,
    pp.strip_multiple_whitespaces,
    pp.strip_punctuation,
    pp.strip_short
          ]

def tokenizer(line):
    """ Applies the following steps in sequence:
        Converts to lower case,
        Strips tags (HTML and others),
        Strips multiple whitespaces,
        Strips punctuation,
        Strips short words(min lenght = 3),
        --------------------------
        :param line: a document
        
        Returns a list of tokens"""
    
    tokens = pp.preprocess_string(line, filters=pp_list)
    return tokens


# In[230]:


get_ipython().run_cell_magic('time', '', "\ntrain_texts = []\n\nfor line in df_downsampled[['description']].fillna(' ').values:\n    train_texts.append(tokenizer(line[0]))#+' '+line[1]))")


# In[231]:


df4 = df_downsampled


# In[232]:


df4['tokens'] = train_texts

df4.head()


# ## Creating Bigrams and Trigrams

# In[233]:


get_ipython().run_cell_magic('time', '', '\nimport gensim\nbigram = gensim.models.Phrases(train_texts)\nbigram_phraser = gensim.models.phrases.Phraser(bigram)\ntokens_ = bigram_phraser[train_texts]\ntrigram = gensim.models.Phrases(tokens_)\ntrigram_phraser = gensim.models.phrases.Phraser(trigram)')


# ### Lemmatizing and Stemming

# In[234]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def process_texts(tokens):
    """Removes stop words, Stemming,
       Lemmatization assuming verb"""
    
    tokens = [token for token in tokens if token not in stop_words]
    tokens = bigram_phraser[tokens]
    tokens = trigram_phraser[tokens]
#     tokens = [stemmer.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    return tokens


# In[235]:


get_ipython().run_cell_magic('time', '', '\nfinal_texts = []\n\nfor line in train_texts:\n    final_texts.append(process_texts(line))')


# In[236]:


df4['final_tokens'] = final_texts
df4.head()


# ## Creating Numerical Codes for Target Classes

# In[239]:


category_codes = {
    'External Fraud': 0,
    'Execution, Delivery and Process Management': 1,
    'Clients, Products and Business Practices' : 2,
    'Information Technology' : 3,
    'Legal and Regulatory' : 4,
    '7. Fraud' : 5,
    'Outsourcing, Vendors, Counterparties and Suppliers' :  6,
    '16. Operations'  : 7,
    'Internal Fraud' : 8,
    'Information Security' :  9    
}

# Category mapping
df4['risk_code'] = df4['risk_taxonomy']
df4 = df4.replace({'risk_code':category_codes})


# In[240]:


df4['input_text'] = df4['final_tokens'].str.join(' ')


# In[241]:


df4.head()


# In[242]:


X_train, X_test, y_train, y_test = train_test_split(df4['input_text'], 
                                                    df4['risk_code'], 
                                                    test_size=0.15, 
                                                    random_state=8)


# ### Creating TF-IDF vectors

# In[243]:


# Parameter election
ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300


# In[244]:


tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
print(features_train.shape)

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
print(features_test.shape)


# ## Supervised Classification Algorithms 

# ## Random Forest

# In[246]:


print(features_train.shape)
print(features_test.shape)
print(labels_train.shape)
print(labels_test.shape)


# In[247]:


#Checking out the available hyperparameters

rf_0 = RandomForestClassifier(random_state = 8)

print('Parameters currently in use:\n')
pprint(rf_0.get_params())


# In[248]:


# Creating a hyperparameter grid for hyperparameter tuning

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)


# In[249]:



rfc = RandomForestClassifier(random_state=8)

random_search = RandomizedSearchCV(estimator=rfc,
                                   param_distributions=random_grid,
                                   n_iter=50,
                                   scoring='accuracy',
                                   cv=3, 
                                   verbose=1, 
                                   random_state=8)

random_search.fit(features_train, labels_train)


# In[250]:


print("The best hyperparameters from Random Search are:")
print(random_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(random_search.best_score_)


# In[151]:


bootstrap = [False]
max_depth = [30, 40, 50]
max_features = ['sqrt']
min_samples_leaf = [1, 2, 4]
min_samples_split = [5, 10, 15]
n_estimators = [800]

param_grid = {
    'bootstrap': bootstrap,
    'max_depth': max_depth,
    'max_features': max_features,
    'min_samples_leaf': min_samples_leaf,
    'min_samples_split': min_samples_split,
    'n_estimators': n_estimators
}

# Create a base model
rfc = RandomForestClassifier(random_state=8)

# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rfc, 
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=cv_sets,
                           verbose=1)

# Fit the grid search to the data
grid_search.fit(features_train, labels_train)


# In[152]:


print("The best hyperparameters from Grid Search are:")
print(grid_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(grid_search.best_score_)


# In[153]:


best_rfc = grid_search.best_estimator_
best_rfc


# In[155]:



best_rfc.fit(features_train, labels_train)
rfc_pred = best_rfc.predict(features_test)
print("The training accuracy is: ")
print(accuracy_score(labels_train, best_rfc.predict(features_train)))


# In[156]:


print("The test accuracy is: ")
print(accuracy_score(labels_test, rfc_pred))


# In[157]:


print("Classification report")
print(classification_report(labels_test,rfc_pred))


# In[159]:



base_model = RandomForestClassifier(random_state = 8)
base_model.fit(features_train, labels_train)
accuracy_score(labels_test, base_model.predict(features_test))


# In[160]:


best_rfc.fit(features_train, labels_train)
accuracy_score(labels_test, best_rfc.predict(features_test))


# In[161]:


d = {
     'Model': 'Random Forest',
     'Training Set Accuracy': accuracy_score(labels_train, best_rfc.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, rfc_pred)
}

df_models_rfc = pd.DataFrame(d, index=[0])


# In[163]:


df_models_rfc


# ## Support Vector Machine 

# In[165]:


from sklearn import svm
svc_0 =svm.SVC(random_state=8)

print('Parameters currently in use:\n')
pprint(svc_0.get_params())


# In[166]:


C = [.0001, .001, .01]

# gamma
gamma = [.0001, .001, .01, .1, 1, 10, 100]

# degree
degree = [1, 2, 3, 4, 5]

# kernel
kernel = ['linear', 'rbf', 'poly']

# probability
probability = [True]

# Create the random grid
random_grid = {'C': C,
              'kernel': kernel,
              'gamma': gamma,
              'degree': degree,
              'probability': probability
             }

pprint(random_grid)


# In[167]:


svc = svm.SVC(random_state=8)


random_search = RandomizedSearchCV(estimator=svc,
                                   param_distributions=random_grid,
                                   n_iter=50,
                                   scoring='accuracy',
                                   cv=3, 
                                   verbose=1, 
                                   random_state=8)


random_search.fit(features_train, labels_train)


# In[168]:


print("The best hyperparameters from Random Search are:")
print(random_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(random_search.best_score_)


# In[169]:


C = [.0001, .001, .01, .1]
degree = [3, 4, 5]
gamma = [1, 10, 100]
probability = [True]

param_grid = [
  {'C': C, 'kernel':['linear'], 'probability':probability},
  {'C': C, 'kernel':['poly'], 'degree':degree, 'probability':probability},
  {'C': C, 'kernel':['rbf'], 'gamma':gamma, 'probability':probability}
]

# Create a base model
svc = svm.SVC(random_state=8)

# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=svc, 
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=cv_sets,
                           verbose=1)

# Fit the grid search to the data
grid_search.fit(features_train, labels_train)


# In[170]:


print(grid_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(grid_search.best_score_)


# In[171]:


best_svc = grid_search.best_estimator_
best_svc


# In[172]:


best_svc.fit(features_train, labels_train)


# In[173]:


svc_pred = best_svc.predict(features_test)


# In[174]:


print("The training accuracy is: ")
print(accuracy_score(labels_train, best_svc.predict(features_train)))


# In[175]:


print("The test accuracy is: ")
print(accuracy_score(labels_test, svc_pred))


# In[176]:


print("Classification report")
print(classification_report(labels_test,svc_pred))


# In[177]:


base_model = svm.SVC(random_state = 8)
base_model.fit(features_train, labels_train)
accuracy_score(labels_test, base_model.predict(features_test))


# In[178]:


best_svc.fit(features_train, labels_train)
accuracy_score(labels_test, best_svc.predict(features_test))


# In[179]:


d = {
     'Model': 'SVM',
     'Training Set Accuracy': accuracy_score(labels_train, best_svc.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, svc_pred)
}

df_models_svc = pd.DataFrame(d, index=[0])


# In[180]:


df_models_svc


# In[ ]:




