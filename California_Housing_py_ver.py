#!/usr/bin/env python
# coding: utf-8

# ## <center> Individual Project Week 11</center>
# 
# ### <i> By: Jacob Oriang Jaroya

# ### We will use the CRISP-DM framework and split the project according to the 5 phases:
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Modeling
# 5. Evaluation

# ### 1.1  Use Case Introduction
# We will work with the California Housing Price dataset made available in a csv file. This dataset was based on data from the 1990 California census.

# ### 1.2  Business Understanding
# 
# The task is to perform and build a model of housing prices in California using the California census data. This data has features such as population, median income, median housing price and so on for each district in California
# 
# The model should learn from this data and be able to predict the median housing price in any district, given all other attributes.

# ### 1.3   Data Understanding
# Data Understanding typically involves the following steps:
# - Determine what data is needed and collect the data if not available
# - Explore data
# - Verify data quality

# #### 1.3.1 Load the dataset and take a quick look at the data structure

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;


# In[2]:


housing=pd.read_csv('housing.csv')


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing["ocean_proximity"].value_counts()


# In[6]:


housing.describe()


# In[7]:


housing.hist(bins=50, figsize=(20,15))
plt.show()


# #### 1.3.2 Explore and visualize the data to gain insights

# #### Visualizing Geographical Data

# In[8]:


# Visualize longitudes and latitudes plotted
housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(15,10));


# In[9]:


# See the pattern clearly by setting the apha=0.1
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, figsize=(15,10))


# In[10]:


# The radius of each circle represents the dis-trict’s population (option s), 
# and the color represents the price (option c).
# Use a predefinedcolor map (option cmap) called jet, which ranges from blue (low values) to red (high prices)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(15,10),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()


# #### Looking for Correlations
# Since the dataset is not too large, we can easily compute the standard correlation coe฀cient (also called Pearson’s r) between every pair of attributes using the corr()method:

# In[11]:


corr_matrix=housing.corr()


# In[12]:


# how much each attribute correlates with the median house value:
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[13]:


# Another way to check for correlation between attributes is to use Pandas’ scatter_matrix function,
# which plots every numerical attribute against every other numerical attribute.

# Focusing on a few promising attributes that seem most correlated with the median housing value
from pandas.plotting import scatter_matrix
attributes=["median_house_value","median_income","total_rooms", "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12,8));


# In[14]:


# The most promising attribute to predict the median house value is the median income, 
# so let’szoom in on their correlation scatterplot:
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, figsize=(15,10));


# In[15]:


# Focusing on a few promising attributes that seem most correlated with the median housing value:
from pandas.plotting import scatter_matrix
attributes=["median_house_value","median_income","total_rooms","housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12,8));


# In[16]:


# The most promising attribute to predict the median house value is the median income, 
# so let’szoom in on their correlation scatterplot:
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, figsize=(15,10))


# #### 1.3.3  Verify data quality
# 
# #### Missing Values

# In[17]:


# We find out which features have missing values, either by applying pandas info() function or the isnull() function:
housing.isnull().sum()


# #### Outliers Values
# - An outlier is a data point that is noticeably different from the rest, that is, an extreme value. 
# - It may be due to variability in the measurement, or it may indicate measurement or execution error.

# In[18]:


# Show the columns in the data set
housing.columns


# In[19]:


# Use the matplotlib.pyplot function boxplot to see outliers:
for col in ['longitude','latitude','housing_median_age','total_rooms', 
            'total_bedrooms','population','households','median_income', 'median_house_value']:
    # exlcude ocean_proximity since not numerical variable
    print(col)
    plt.boxplot(housing[col])
    plt.show()


# #### 2.1   Data Preparation
# 
# #### This stage of CRISP-DM process can be broken into the following steps:
# - Split data into train and test sets
# - Clean data
# - Feature Engineering (Generate new variables & variable selection)
# - Feature Scaling

# #### 2.1.1  Split data into train and test sets

# - Scikit-Learn provides a few functions to split data sets into multiple subsets in various ways. 
# - The simplest function is train_test_split
# - The dataset will be split into 80% of training and the remaining 20% will be for testing. 
# - The split is performed randomly, meaning, the node randomly selects 80% of the data. 
# - The random_state parameter is set to a random number, so that the function always generates the same train and test sets.

# In[20]:


from sklearn.model_selection import train_test_split
train, test=train_test_split(housing, test_size=0.2, random_state=20)


# In[21]:


# Checking the size of the training and test sets with the shape() function:
train.shape, test.shape


# In[22]:


train.isnull().sum()


# In[23]:


test.isnull().sum()


# #### 2.1.2  Clean data
# 
# #### To clean the data, we need to take care of: 
# - the missing values, 
# - features containing non-numeric values 
# - and in case of outliers, we decide whether to keep them or remove them.

# #### 2.1.3  Handling Missing Values
# 
# #### There are 3 approaches to deal with missing values:
# - Delete missing values
# - Impute for missing values
# - Apply a ML model to predict the missing values

# In[24]:


# We use the SimpleImputer class for missing values
from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(strategy="median")


# In[25]:


# Since the median can only be computed on numerical attributes, 
# we need to create a copy of the data without the text attribute "ocean_proximity":
train_num=train.drop("ocean_proximity", axis=1)


# In[26]:


train_num.head()


# In[27]:


# Now we fit the imputer instance to the training data using the fit() method:
imputer.fit(train_num)


# In[28]:


# The imputer has simply computed the median of each attribute and stored the result in its 'statistics_' instance variable.
imputer.statistics_


# In[29]:


# Get same values...
train_num.median().values


# In[30]:


# Now we use this “trained” imputer to transform the training set by replacing missing values by the learned medians:
X=imputer.transform(train_num)


# In[31]:


# The result is a plain NumPy array containing the transformed features. 
# We put it back into a Pandas DataFrame as follows:
train_tr=pd.DataFrame(X, columns=train_num.columns)


# In[32]:


train_tr


# In[33]:


train_tr.isnull().sum()


# #### 2.1.4  Handling Categorical Features
# - Most Machine learning models require all input and output variables to be numeric. 
# - This means that if data contains categorical data, we must encode it to numbers before we can fit and evaluate the model.

# #### As for missing values, we first explore what options we have to transform categorical values to numbers. 
# There are 3 common approaches that we can follow:
# 1. Ordinal Encoding
#     - In ordinal encoding, each unique category value is assigned an integer value.
#     - For example, “male” is 1, “female” is 2, or “good” is 1, “medium” is 2 and “bad” is 3
# 2. One-Hot Encoding
#     - One-Hot Encoding is the most common way to deal with non-ordinal cate-gorical data. 
#     - It consists of creating an additional feature for each category of the categorical featureand mark each observation belonging (Value=1) or not (Value=0) to that group.
# 3. Target Encoding
#     - A lesser known, but very effective way of handling categorical variables, is Target Encoding. 
#     - It consists of substituting each category in a categorical feature with the average response in the target variable.

# In[34]:


#  The attribute 'ocean_proximity' is a categorical feature with categories: <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND.
# Since the number of categories are moderate, i.e., 5, the better approach would be one-hot encoding.
train_cat=train[["ocean_proximity"]]
train_cat.head(10)


# In[35]:


# Scikit-Learn provides a OneHotEncoder class to convert categorical values into one-hot vectors:
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
train_cat_1hot = cat_encoder.fit_transform(train_cat)
train_cat_1hot


# In[36]:


# We can get the list of categories using the encoder’s categories_instance variable
cat_encoder.categories_


# #### 2.1.5  Feature engineering
# Feature engineering consists of creation, transformation, extraction, and selection of features.

# #### The total number of rooms in a district is not very useful if you don’t know how many householdsthere are. 
# 1. What we really want is the number of rooms per household. 
# 2. Similarly, the total numberof bedrooms by itself is not very useful: we want to compare it to the number of rooms.
# 3. And the population per household also seems like an interesting attribute combination to look at.

# In[37]:


# Let’s create these new attributes:
train["rooms_per_household"] = train["total_rooms"]/train["households"]
train["bedrooms_per_room"] = train["total_bedrooms"]/train["total_rooms"]
train["population_per_household"] = train["population"]/train["households"]


# In[38]:


train.head()


# In[39]:


# Now let’s look at the correlation matrix again:
corr_matrix=train.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)


# #### 2.1.6    Feature Scaling
#  Note that scaling the target values is generally not required.
# #### There are two common ways to get all attributes to have the same scale: 
# - min-max scaling and
# - standardization
# 
# #### Remark: 
# As with all the transformations, it is important to fit the scalers to the training data only, not to the full dataset (including the test set). Only then can you use them to transform the training set and the test set (and new data).

# #### Transformation Pipelines
# There are many data transformation steps that need to be executed in the right order. Fortunately, Scikit-Learn provides the Pipeline class to help with such sequences of transformations. Before we apply it, we need first to split the data into input and outputs:

# In[40]:


train_labels = train['median_house_value'].copy()

# drop the labels from the train set via the pandas drop function:

train = train.drop('median_house_value', axis=1) # axis=1 means that median_house_value should be dropped column wise, 
# meaning, the whole column will be dropped

train_num = train.drop("ocean_proximity", axis=1)#  re-do since the additional columns rooms_per_household, 
# bedrooms_per_room and population_per_household where added

train_labels.head()


# In[41]:


train.head()


# In this case, the pipeline has two steps:
# 1. <b>imputer</b>, which imputes missing values in the data using the median of the remaining values. This is done using an instance of the <b>SimpleImputer class</b> with the strategy parameter set to "median".
# 2. <b>std_scaler</b>, which standardizes the data by subtracting the mean and dividing by the standard deviation. This is done using an instance of the <b>StandardScaler class.</b>
# <p>
#     Once the pipeline is created, we can fit it to the <b><i>training</i></b> data using the <b><i>fit</i></b> method, and then use it to transform new data or make predictions using the <b>transform or predict methods</b>, respectively.
# 

# In[42]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('std_scaler', StandardScaler())])

train_num_tr = num_pipeline.fit_transform(train_num)


#  Here is how this works: 
# - first we import the ColumnTransformer class, 
# - next we get the list of numerical column names and the list of categorical column names, and we construct a ColumnTransformer.
# #### The constructor requires a list of tuples, where each tuple contains a name, a transformer and alist of names (or indices) of columns that the transformer should be applied to.
# - In this example, we specify that the numerical columns should be transformed using the num_pipeline that we defined earlier
# - and the categorical columns should be transformed using a OneHotEncoder. 
# #### Finally, weapply this ColumnTransformer to the housing data: 
# - it applies each transformer to the appropriate columns and 
# - concatenates the outputs along the second axis (the transformers must return the same number of rows)

# In[43]:


from sklearn.compose import ColumnTransformer

num_attribs = list(train_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs)])

train_prepared = full_pipeline.fit_transform(train)


# #### 3.1  Modeling
# At this stage, are now ready to select and train a Machine Learning model.
# #### 3.1.1   Training and Evaluating on the Training Set
# We are going to train three different models and compare their performance on the training set.
# <p><b>Remark:</b></p>
#     The overall goal is to train a model that performs well on test set. But analyzing the performance of a model on the training set is also important; it gives you a first glance on the overall performance. A model that does not perform well on training set, cannot perform well onthe test set either.
# 
# #### (1) Linear Regression model
# Let us first train a Linear Regression model

# In[44]:


train_prepared.shape


# In[45]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_prepared, train_labels)


# In[46]:


# You now have a working Linear Regression model. 
# Let’s try it out on a few instances fromthe training set:

some_data = train.iloc[:5]
some_labels = train_labels.iloc[:5]

# transform / prepare some data
some_data_prepared = full_pipeline.transform(some_data)

# make predictions
print("Predictions:", lin_reg.predict(some_data_prepared))


# In[47]:


print("Labels:",list(some_labels))


# It works, although the predictions are not exactly accurate (e.g., the first prediction is off by close to 40%!). Let’s measure this regression model’s RMSE on the whole training set using Scikit-Learn’s mean_squared_error function:

# In[48]:


from sklearn.metrics import mean_squared_error

train_predictions = lin_reg.predict(train_prepared)
lin_mse = mean_squared_error(train_labels, train_predictions)
lin_rmse = np.sqrt(lin_mse)

lin_rmse


# - The above result is clearly not a great score:  most district’s <b>median_housing_value</b> range between 120,000 and 265,000 (as seen from the box plot previously), so a <b>typical prediction error of 67,593 is not very satisfying.</b>
# - Let us first try a more complex model to see how it does.

# #### (2) Decision Tree model
# Next we train a Decision Tree model. This model is capable off inding complex non-linear relationships in the data.

# In[49]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_prepared, train_labels)


# In[159]:


# Now that the model is trained, let’s evaluate it on the training set:

train_predictions_tree = tree_reg.predict(train_prepared)
tree_mse = mean_squared_error(train_labels, train_predictions_tree)
tree_rmse = np.sqrt(tree_mse)

tree_rmse


# #### (2.1) Better Evaluation Using Cross-Validation
# One way to evaluate the Decision Tree model would be to use the <b>train_test_split</b> function to split the training set into a smaller training set and a validation set, then train your models against the smaller training set and evaluate them against the validation set.
# 
# <p> A great alternative is to use <b><i>Scikit-Learn’s K-fold cross-validation</b></i> feature. The following code randomly splits the training set into 10 distinct subsets called folds, then it trains and evaluates the Decision Tree model 10 times, picking a different fold for evaluation every time and training on the other 9 folds. The result is an array containing the 10 evaluation scores:

# In[160]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, train_prepared, train_labels,scoring = "neg_mean_squared_error", cv=10)

tree_rmse_scores=np.sqrt(-scores)


# In[161]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


# #### Remark: 
# The Decision Tree model here above is a perfect example of overfitting. 
# 1. On the training set, the model performed perfectly, while on the validation sets, it did not. 
# 2. This can be generalized as follows:
#     - if a model performs better on the training set than on the validation sets, that is, with Cross-Validation, then the model is probably overfitting.
# 
# Let’s compute the same scores for the Linear Regression model just to be sure:

# In[162]:


lin_scores = cross_val_score(lin_reg, train_prepared, train_labels,scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)


# The Decision Tree model is overfitting so badly that it performs worse than the LinearRegression model.

# #### (3) Random Forest model
# This model is an example of an Ensemble Learning.
# 
# - If we aggregate the predictions of a group of models,we will often get better predictions than with the best individual predictor. 
# - A group of predictors is called an <b>ensemble</b>; thus, this technique is called <b>Ensemble Learning</b>, and an Ensemble Learning algorithm is called an <b>Ensemble method</b>.
# <p>
# - For example, we can train a group of Decision Tree models, each on a different random subset of the training set. 
# - To make predictions, you just obtain the predictions of all individual trees, then compute the averages of those predictions to obtain the final predictions. 
# - Such an ensembleof Decision Trees is called a Random Forest, and despite its simplicity, this is a powerful Machine Learning algorithm.

# In[163]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(train_prepared, train_labels)

train_predictions_RF = forest_reg.predict(train_prepared)
forest_mse = mean_squared_error(train_labels, train_predictions_RF)

forest_rmse = np.sqrt(forest_mse)

forest_rmse


# In[164]:


forest_scores = cross_val_score(forest_reg, train_prepared, train_labels,scoring = "neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# #### (3.1) Fine-Tune Your Model
#     We will explore the Random Forest model in more depth and see how we can fine-tune it. Byfine-tuning we mean hyperparameter optimization
# 
# #### Grid Search
#         Scikit-Learn’s GridSearchCV will evaluate all the possible combinations of hyperparameter values, 
#         using cross-validation. 
#         All you need to do is tell it which hyperparameters you want it to experiment with, and what values to try out.
#     
#     The following code searches for the best combination of hyperparameter values for the RandomForestRegressor:

# In[165]:


from sklearn.model_selection import GridSearchCV

param_grid = [ {'n_estimators': [3,10,30],'max_features': [2,4,6,8]}, 
              {'bootstrap': [False],'n_estimators': [3,10],'max_features': [2,3,4]},]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(train_prepared, train_labels)


# In[166]:


grid_search.best_params_ # our result {'max_features': 6, 'n_estimators': 30}

# Since 8 and 30 are the maximum values that were evaluated, you should
# probably try searching again with higher values, since the score may continue to improve.


# In[167]:


# You can also get the best estimator directly:
grid_search.best_estimator_


# In[168]:


# The evaluation scores are also available:
cvres=grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[170]:


# The evaluation the performance of Stochastic Gradient Descent modelwith training data:
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(train_labels, train_predictions_RF)
mae = mean_absolute_error(train_labels, train_predictions_RF)
r2_RF = r2_score(train_labels, train_predictions_RF)

print("Mean Squared Error: ", mse)
print("Mean Absolute Error: ", mae)
print("R2 Score: ", r2_RF)


# In[171]:


#R-squared value as a percentage
print( f"The accuracy of the Random Forest model on Training Data stands at: {r2_RF:.1%}" )


# ## -------------------------------------------------------------------------------------------

# ## [4] Stochastic Gradient Descent model

#  SGDRegressor is imported from the linear_model module of scikit-learn. 
#  
# - alpha:
# 
#     This parameter represents the regularization strength. Regularization is a technique used to prevent overfitting in the model by adding a penalty term to the loss function. The value of alpha determines the magnitude of the penalty term. A smaller value of alpha implies a smaller penalty and a more complex model, while a larger value of alpha implies a stronger penalty and a simpler model.
# 
# - loss: 
# 
#     This parameter determines the loss function used by the model. The loss parameter can be set to 'squared_loss', 'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive' to use the corresponding loss function. The choice of loss function can have a significant impact on the performance of the model, and finding the optimal value for this hyperparameter is often an important part of the model tuning process.
# 
# - penalty: 
# 
#     This parameter determines the type of regularization used by the model. The penalty parameter can be set to 'l2', 'l1', 'elasticnet', or 'none' to use the corresponding regularization method. The choice of regularization method can have a significant impact on the performance of the model, and finding the optimal value for this hyperparameter is often an important part of the model tuning process. In your code, penalty='elasticnet' specifies that the Elastic Net regularization method is used. This method combines both L1 and L2 regularization penalties.

# In[79]:


# Import the necessary library
from sklearn.linear_model import SGDRegressor


# In[149]:


# Create an SGD model and fit it on the training data:

# Initialize the SGDRegressor
import warnings
warnings.filterwarnings("ignore")

sgd = SGDRegressor(alpha=0.1, loss='squared_loss',penalty='elasticnet')
sgd.fit(train_prepared, train_labels)


# In[150]:


# Predict on the training data and evaluate the model:

train_predictions_SGD = sgd.predict(train_prepared)
SGD_mse = mean_squared_error(train_labels, train_predictions_SGD)

SGD_rmse = np.sqrt(SGD_mse)

SGD_rmse


# In[151]:


SGD_scores = cross_val_score(sgd, train_prepared, train_labels,scoring = "neg_mean_squared_error", cv=10)
SGD_rmse_scores = np.sqrt(-SGD_scores)
display_scores(SGD_rmse_scores)


# #### (4.1) Fine-Tune Your Model
#     We will explore the Stochastic Gradient Descent model in more depth and see how we can fine-tune it. By fine-tuning we mean hyperparameter optimization
# 
# #### Grid Search
#         Scikit-Learn’s GridSearchCV will evaluate all the possible combinations of hyperparameter values, 
#         using cross-validation. 
#         All you need to do is tell it which hyperparameters you want it to experiment with, and what values to try out.
#     
#     The following code searches for the best combination of hyperparameter values for the Stochastic Gradient Descent:

# In[152]:


from sklearn.model_selection import GridSearchCV
param_grid = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
               'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
               'penalty': ['l2', 'l1', 'elasticnet']}]

sgd = SGDRegressor()
grid_search_SGD = GridSearchCV(sgd, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search_SGD.fit(train_prepared, train_labels)


# In[153]:


grid_search_SGD.best_params_


# In[154]:


# We can also get the best estimator directly:
grid_search_SGD.best_estimator_


# To evaluate the performance of the SGDRegressor model, we can use various metrics such as:
# - mean squared error (MSE), 
# - mean absolute error (MAE), 
# - R2 score, etc. 
# 
# We can use the mean_squared_error, mean_absolute_error, and r2_score functions from the sklearn.metrics module to calculate these metrics as shown below:

# In[172]:


# The evaluation the performance of Stochastic Gradient Descent model with training data:

mse = mean_squared_error(train_labels, train_predictions_SGD)
mae = mean_absolute_error(train_labels, train_predictions_SGD)
r2_SGD = r2_score(train_labels, train_predictions_SGD)

print("Mean Squared Error: ", mse)
print("Mean Absolute Error: ", mae)
print("R2 Score: ", r2_SGD)


# In[174]:


#R-squared value as a percentage
print( f"The accuracy of the Stochastic Gradient Descent model on Training Data stands at: {r2_SGD:.1%}" )


# ### Time to save our 2 Models:
#     1. Random Forest
#     2. Stochastic Gradient Descent

# Pickle is the standard way of serializing objects in Python. 
# 
# We can use the pickle operation to serialize our machine learning algorithms and save the serialized format to a file:

# In[175]:


import pickle 
# from sklearn.externals import joblib

forest_model='forest_housing_model.pkl'

#joblib.dump(grid_search.best_estimator_,filename)

forest_model='forest_housing_model.sav'

pickle.dump(grid_search.best_estimator_,open(forest_model,'wb'))


stochasticGD_model='stochasticGD_housing_model.pkl'

#joblib.dump(grid_search.best_estimator_,filename)

stochasticGD_model='stochasticGD_housing_model.sav'

pickle.dump(grid_search_SGD.best_estimator_,open(stochasticGD_model,'wb'))


# # <center> Working with Test Data

# #### 1.  Handling Missing Values
# 
# #### There are 3 approaches to deal with missing values:
# - Delete missing values
# - Impute for missing values
# - Apply a ML model to predict the missing values

# In[177]:


test.isnull().sum()


# In[178]:


# We use the SimpleImputer class for missing values
from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(strategy="median")


# In[179]:


# Since the median can only be computed on numerical attributes, 
# we need to create a copy of the data without the text attribute "ocean_proximity":
test_num=test.drop("ocean_proximity", axis=1)


# In[180]:


test_num.head()


# In[181]:


# Now we fit the imputer instance to the testing data using the fit() method:
imputer.fit(test_num)


# In[182]:


# The imputer has simply computed the median of each attribute and stored the result in its 'statistics_' instance variable.
imputer.statistics_


# In[183]:


# Get same values...
test_num.median().values


# In[184]:


# Now we use this “trained” imputer to transform the testing set by replacing missing values by the learned medians:
X=imputer.transform(test_num)


# In[185]:


# The result is a plain NumPy array containing the transformed features. 
# We put it back into a Pandas DataFrame as follows:
test_tr=pd.DataFrame(X, columns=test_num.columns)


# In[186]:


test_tr


# In[187]:


test_tr.isnull().sum()


# #### 2. Handling Categorical Features
# - Most Machine learning models require all input and output variables to be numeric. 
# - This means that if data contains categorical data, we must encode it to numbers before we can fit and evaluate the model.

# #### As for missing values, we first explore what options we have to transform categorical values to numbers. 
# There are 3 common approaches that we can follow:
# 1. Ordinal Encoding
#     - In ordinal encoding, each unique category value is assigned an integer value.
#     - For example, “male” is 1, “female” is 2, or “good” is 1, “medium” is 2 and “bad” is 3
# 2. One-Hot Encoding
#     - One-Hot Encoding is the most common way to deal with non-ordinal cate-gorical data. 
#     - It consists of creating an additional feature for each category of the categorical featureand mark each observation belonging (Value=1) or not (Value=0) to that group.
# 3. Target Encoding
#     - A lesser known, but very effective way of handling categorical variables, is Target Encoding. 
#     - It consists of substituting each category in a categorical feature with the average response in the target variable.

# In[188]:


# The attribute 'ocean_proximity' is a categorical feature with categories: <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND.
# Since the number of categories are moderate, i.e., 5, the better approach would be one-hot encoding.
test_cat=test[["ocean_proximity"]]
test_cat.head(10)


# In[189]:


# Scikit-Learn provides a OneHotEncoder class to convert categorical values into one-hot vectors:
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
test_cat_1hot = cat_encoder.fit_transform(test_cat)
test_cat_1hot


# In[190]:


# We can get the list of categories using the encoder’s categories_instance variable
cat_encoder.categories_


# #### 3.  Feature engineering
# Feature engineering consists of creation, transformation, extraction, and selection of features.

# #### The total number of rooms in a district is not very useful if you don’t know how many householdsthere are. 
# 1. What we really want is the number of rooms per household. 
# 2. Similarly, the total numberof bedrooms by itself is not very useful: we want to compare it to the number of rooms.
# 3. And the population per household also seems like an interesting attribute combination to look at.

# In[192]:


# Let’s create these new attributes:
test["rooms_per_household"] = test["total_rooms"]/test["households"]
test["bedrooms_per_room"] = test["total_bedrooms"]/test["total_rooms"]
test["population_per_household"] = test["population"]/test["households"]


# In[193]:


test.head()


# In[194]:


# Now let’s look at the correlation matrix again:
corr_matrix=test.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)


# #### 4.    Feature Scaling
#  Note that scaling the target values is generally not required.
# #### There are two common ways to get all attributes to have the same scale: 
# - min-max scaling and
# - standardization
# 
# #### Remark: 
# As with all the transformations, it is important to fit the scalers to the training data only, not to the full dataset (including the test set). Only then can you use them to transform the training set and the test set (and new data).

# #### Transformation Pipelines
# There are many data transformation steps that need to be executed in the right order. Fortunately, Scikit-Learn provides the Pipeline class to help with such sequences of transformations. Before we apply it, we need first to split the data into input and outputs:

# In[195]:


test_labels = test['median_house_value'].copy()

# drop the labels from the test set via the pandas drop function:

test = test.drop('median_house_value', axis=1) # axis=1 means that median_house_value should be dropped column wise, 
# meaning, the whole column will be dropped

test_num = test.drop("ocean_proximity", axis=1)#  re-do since the additional columns rooms_per_household, 
# bedrooms_per_room and population_per_household where added

test_labels.head()


# In[196]:


test.head()


# In[197]:


test_num.head()


# In this case, the pipeline has two steps:
# 1. <b>imputer</b>, which imputes missing values in the data using the median of the remaining values. This is done using an instance of the <b>SimpleImputer class</b> with the strategy parameter set to "median".
# 2. <b>std_scaler</b>, which standardizes the data by subtracting the mean and dividing by the standard deviation. This is done using an instance of the <b>StandardScaler class.</b>
# <p>
#     Once the pipeline is created, we can fit it to the <b><i>testing</i></b> data using the <b><i>fit</i></b> method, and then use it to transform new data or make predictions using the <b>transform or predict methods</b>, respectively.

# In[198]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('std_scaler', StandardScaler())])

test_num_te = num_pipeline.fit_transform(test_num)


#  Here is how this works: 
# - first we import the ColumnTransformer class, 
# - next we get the list of numerical column names and the list of categorical column names, and we construct a ColumnTransformer.
# #### The constructor requires a list of tuples, where each tuple contains a name, a transformer and a list of names (or indices) of columns that the transformer should be applied to.
# - In this example, we specify that the numerical columns should be transformed using the num_pipeline that we defined earlier
# - and the categorical columns should be transformed using a OneHotEncoder. 
# #### Finally, we apply this ColumnTransformer to the housing data: 
# - it applies each transformer to the appropriate columns and 
# - concatenates the outputs along the second axis (the transformers must return the same number of rows)

# In[200]:


test_num.shape


# In[201]:


from sklearn.compose import ColumnTransformer

num_attribs = list(test_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs)])

test_prepared = full_pipeline.fit_transform(test)


# In[202]:


pd.DataFrame(test_prepared)


# #### 5.  Modeling
# We are now ready to test our model.

# In[203]:


test_prepared.shape


# ### Loading the saved Models

# In[204]:


# Load the saved models:
import joblib

forest_model='forest_housing_model.sav'
model_FR = joblib.load(forest_model)

stochasticGD_model='stochasticGD_housing_model.sav'
model_SGD = joblib.load(stochasticGD_model)


# #### 6 Predict Values
# We make predictions with our saved models based on the test data

# In[205]:


# Predict using the loaded models:

y_pred_FR = model_FR.predict(test_prepared)
y_pred_SGD = model_SGD.predict(test_prepared)


# In[207]:


# Displaying the first 5 predictions using the Random Forest Model:

print("Prediction from RF Model:", y_pred_FR[0:5])


# In[208]:


# Comparing with the first 5 actual values:

print("Actual values:\n", test_labels[0:5])


# In[209]:


# Displaying the first 5 predictions using the Stochastic Gradient Descent model:

print("Prediction from SGD Model:", y_pred_SGD[0:5])


# #### Plotting a Scatter plot:
# - The accuracy line in the plot indicates how well the predicted values match the actual values. 
# - The accuracy line is a <b>45-degree line from the origin to the upper-right corner of the plot</b>, where the predicted values perfectly match the actual values.
# 
# The [min_val, max_val] values represent the x-axis and y-axis limits of the plot, and [min_val, max_val] is used as the x and y values to plot a line from the lower-left corner to the upper-right corner of the plot.

# In[210]:


# RANDOM FOREST PLOT
# Plot the test data vs. predicted data

plt.scatter(test_labels, y_pred_FR, color='blue')

# plot the accuracy line
min_val = min(min(test_labels), min(y_pred_FR))
max_val = max(max(test_labels), max(y_pred_FR))
plt.plot([min_val, max_val], [min_val, max_val], '--', color='black') 

# show the plot
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted Values vs Actual - Random Forest Model")


# In[212]:


# STOCHASTIC GRADIENT DESCENT
# Plot the test data vs. predicted data

plt.scatter(test_labels, y_pred_SGD, color='red')

# plot the accuracy line
min_val = min(min(test_labels), min(y_pred_SGD))
max_val = max(max(test_labels), max(y_pred_SGD))
plt.plot([min_val, max_val], [min_val, max_val], '--', color='black') 

# show the plot
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted Values vs Actual - Stochastic Gradient Descent Model")


# ### Observation:
# The 45-degree accuracy line shows where the predicted values perfectly match the actual values. 
# 
# - It is very evident that a GOOD fit can be observed in Random Forest Scatter plot where most ponts are along the 45-degree line.

# ####  Evaluating the 2 Models:
# We shall check for errors based on the values the 2 models predicted and the actual values from the test dataset

# 1. Mean Absolute Error (MAE): It is the average of the absolute differences between the actual and predicted values.
# 
# 2. Mean Squared Error (MSE): It is the average of the squares of the differences between the actual and predicted values.
# 
# 3. Root Mean Squared Error (RMSE): It is the square root of the mean squared error.
# 
# 4. R-squared: It is a statistical measure that represents the proportion of variance in the target variable that can be explained by the predictors.

# In[213]:


# Calculate mean absolute error
mae_FR = mean_absolute_error(test_labels, y_pred_FR)
print("Mean Absolute Error from Random Forest:", mae_FR)
mae_SGD = mean_absolute_error(test_labels, y_pred_SGD)
print("Mean Absolute Error from Stochastic Gradient Descent:", mae_SGD)
print()

# Calculate mean squared error
mse_FR = mean_squared_error(test_labels, y_pred_FR)
print("Mean Squared Error from Random Forest:", mse_FR)
mse_SGD = mean_squared_error(test_labels, y_pred_SGD)
print("Mean Squared Error from Stochastic Gradient Descent:", mse_SGD)
print()

# Calculate root mean squared error
rmse_FR = np.sqrt(mse_FR)
print("Root Mean Squared Error from Random Forest:", rmse_FR)
rmse_SGD = np.sqrt(mse_SGD)
print("Root Mean Squared Error from Stochastic Gradient Descent:", rmse_SGD)
print()

# Calculate R-squared score
r2_FR = r2_score(test_labels, y_pred_FR)
print("R-squared from Random Forest:", r2_FR)
r2_SGD = r2_score(test_labels, y_pred_SGD)
print("R-squared from Stochastic Gradient Descent:", r2_SGD)


# In[218]:


#R-squared value as a percentage
print( f"The accuracy of the Random Forest model on Test Data stands at:\t\t\t{r2_FR:.1%}" )
print( f"The accuracy of the Stochastic Gradient Descent model on Test Data stands at:\t{r2_SGD:.1%}" )


# ### Conclusion:
# 1. As both RMSE and MAE values are negatively oriented scores, we see that:
#     - Mean Absolute Error from Random Forest = 44072.88502906976
#     - Mean Absolute Error from Stochastic Gradient Descent = 66084.29388426962
#     
#     <center>AND</center>
#     - Root Mean Squared Error from Random Forest = 61201.16580306076
#     - Root Mean Squared Error from Stochastic Gradient Descent= 85841.16475275194
#     
# 2. The accuracy score are:
#     - The accuracy of the Random Forest model on Test Data stands at = 73.1%
#     - The accuracy of the Stochastic Gradient Descent model on Test Data stands at = 47.2%
# 
# <b> Therefore Random Forest is performing better than Stochastic Gradient Descent model</b>

# #### For Regression problems with many features, some popular alternatives to Linear Regression, Decision Tree, Random Forest and Stochastic Gradient Descent include:
# 1.	Gradient Boosting Regression (e.g. XGBoost or LightGBM)
# 2.	Support Vector Regression (SVR)
# 3.	Neural Networks (e.g. Multi-layer Perceptron)
# 4.	Bayesian Ridge Regression
# 5.	Lasso Regression
# 6.	Ridge Regression
# 

# In[ ]:




