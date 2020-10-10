# What factors contribute to higher rates of adolescent maternal mortality long-term?

# Machine learning methods are becoming increasingly relevant as data becomes more readily available.
# These methods enable researchers to expand on their general understanding of the complex dynamic changes
# in indicators with a significant number of interrelated factors. Additionally, it is possible to make the
# results, like external factors affecting the incidence of adolescent maternal mortality, interpretable from
# a clinical point of view.
# 
# The following methods consist of: Random Forest Classifier and Logistic Regression. For the interpretation
# of differences between individual parameters of the mean age of maternal mortality two classes predicted failure
# (1 - region mean age is below the country mean) verses prediction success (0 - reagion mean age is above the
# country mean), the problem of classification via using the decision tree can be solved. The target is prediction
# failure class.
# 
# **The primary purpose of this study was to assess the rates of adolescent maternal mortality within Mexico, and
# calculate the risk probability of an adolescent maternal mortality by region in Mexico based on some of the top
# features contributing to maternal mortality.**

# Import the relevant modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Hyperparameters
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz

# ROC Curve
from sklearn.metrics import roc_curve

# Scale Data
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Open merged materna_mortal_factors dataset
#get_ipython().run_line_magic('store', '-r metro_gdp_mortality')
data = metro_gdp_mortality

# Dataset Target Variable: Above(0) or Below(1) MEX μ

# Assess potential correlation of various factors within each Region
# - Create a correlation dataframe
# - Plot the correlation dataframe on a sns heatmap:
#     - Cells that are in green show positive correlation
#     - Cells that are in red show negative correlation

# Create a correlation dataframe
feature_corr = data.corr()

# Plot a correlation heatmap
sns.heatmap(feature_corr, square=True, cmap='RdYlGn')
data.corr()

# *Preliminary Observations of Interest*:
# - **Positive (+) Correlation** *(as 'X' increases, so does 'Y')*
#     - State Population and State GDP
#     - State Population and μ Local Community Size
#     - μ Region Education and μ Local Community Size
# - **Negative (-) Correlation** *(as 'X' increases, 'Y' decreases)*
#     - State Population and Above(0)/Below(1) MEX μ *(age maternal mortality)*
#     - State GDP and μ Region Education Level
#     - Increase(0)/Not(1) GDP 2010-15 and μ Presence(0)/Not(1) of Med Assist
#     - Above(0)/Below(1) MEX μ *(age maternal mortality)* and μ Presence(0)/Not(1) of Med Assist

# ## Prepare Data for Logisitic Regression Machine Learning Model

# #### Purpose for Changing all Categorical Strings to a Numeric Value: 
# - Machine Learning models will ignore string values (strings have no statistical value unless added)
# - Numeric values are comparable therefore string values should be categorically changed to numbers
# - This is how you compare a string value to a numeric value that the model can use

# Convert Column value strings to a numeric value
for i, column in enumerate(list([str(d) for d in data.dtypes])):
    if column == "object":
        data[data.columns[i]] = data[data.columns[i]].fillna(data[data.columns[i]].mode())
        data[data.columns[i]] = data[data.columns[i]].astype("category").cat.codes
    else:
        data[data.columns[i]] = data[data.columns[i]].fillna(data[data.columns[i]].median())

# Basic Logistic Regression:
# * **Target Variable: Above(0)/Not(1) MeEX μ *(age maternal mortality)***
# * Split the data into a training and test (hold-out) set
# * Train on the training set, and test for accuracy on the testing set

# Create a Logistic Regression Incidence 
clf_log = LogisticRegression(random_state=42, solver='lbfgs')

# Test-Train-Split:
# - X = copy of all features and the response variable of dataset
# - y = all features within dataset *excluding the response (target) variable*
# - test_size = represents the proportion of the dataset (as a percentage) to include in the test split
# - random_state = the seed used by the random number generator

# Entire dataset (even with response variable)
X = data.copy()

# The response variable
y = data.copy().pop('Above(0) or Below(1) MEX μ')

# Create train and test data sets with train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=42)
len(X_train), len(X_test), len(y_train), len(y_test)


# Round 1: Unscaled Data

# Random Forest Classifier: Visualize Data and Determine How Features Interact
rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

# Random Forest Classifier with Unscaled Data
rfc_unscaled = rfc.fit(X_train, y_train)

# Weighted importances of the variables
{X.columns[i]: weight for i, weight in enumerate(rfc_unscaled.feature_importances_)}

# Extract single tree
estimator = rfc_unscaled.estimators_[5]

# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = X.columns,
                class_names = 'Above(0)/Not(1) Mexico μ Age Maternal Mortality',
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')


# *Observations*: It appears that the primary feature that predicts the target variable are State Population (2015),
# however, this may be due to the lack of datapoint within each feature. The unscaled dataset will be assessed first,
# but increased model accuracy will most likely be obtainable by scaling the data.

# Fit Logistic Regression with Unscaled Data
# Fit clf_log to training data (unscaled)
clf_log_unscaled = clf_log.fit(X_train, y_train)

# Examine the coefficients -each parameter has an effect on the target variable result
list(np.transpose(clf_log_unscaled.coef_))

# Calculate the class probablity -returns the probability of model's ability to predict a likely value of the features
probability = clf_log_unscaled.predict_proba(X_test)

# Predict the model -assess how well the model predicts unseen data
predict = clf_log_unscaled.predict(X_test)

# Compute classification report (https://en.wikipedia.org/wiki/Precision_and_recall)
class_report = classification_report(y_test, predict)

# Classification Report
# - **Precision**: 
#     - positive predictive value
#     - total number predicted correctly
# - **Recall**: 
#     - model sensitivity 
#     - the fraction of the relevant documents that are successfully retrieved
# - **F1-score**:
#     - single measurement of system
#     - provide a single measurement for a system

# **Classification Report Conclusion for Unscaled Data**: The Logistic Regression model produced a 0% percision,
# recall, and F1-score for all instances of '1' for the target variable, 'Above(0)/Not(1) MEX μ' *(age maternal
# mortality)* while producing a 57% percision, 80% recall, and 67% F1-score for instances of '0' for the target
# variable. This means that the model, while marginally accurate for detecting Regions with a μ Age Maternal
# Mortality that is above the μ Age Maternal Mortality in Mexico, is not a fully-encompassing model since it
# cannot accurately detect the when a region is below the country mean.

# Compute the confusion_matrix to evaluate the accuracy of a classification
conf_matrix = confusion_matrix(y_test, predict)

# Plot confusion_matrix
plt.figure(figsize = (10,7))
sns.heatmap(conf_matrix, annot=True)

# Compute predicted probabilities: y_pred_prob
y_pred_prob = clf_log_unscaled.predict_proba(X_test)[:,1]

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(clf_log, X, y, scoring='roc_auc', cv=5)

#  Visualize Logistic Regression Model Accuracy with ROC curve
# Compute predicted probabilities: y_pred_prob
y_pred_prob = clf_log_unscaled.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')


# Tuning the Model
# As stated above, this dataset is skewed in terms of number prediction (ie. cannot predict case 1 but can
# predict case 0). Most likely, this problem is due to the small size of the dataset; therefore, the same
# mechanism will be repeated as above but with scaled data.

# Round 2: Scaled Data
# Scale Data for Logistic Regression with Standard Scaler and Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('log', LogisticRegression(random_state=42, solver='lbfgs'))]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Logistic Regression with Scaled Data

# Fit the pipeline to the training set: clf_log_scaled
clf_log_scaled = pipeline.fit(X_train, y_train)

# Calculate the class probablity -returns the probability of model's ability to predict a likely value of the features
probability = clf_log_scaled.predict_proba(X_test)

# Predict the model -assess how well the model predicts unseen data
predict_scaled = clf_log_scaled.predict(X_test)

# Compute classification report (https://en.wikipedia.org/wiki/Precision_and_recall)
class_report = classification_report(y_test, predict_scaled)

# **Classification Report Conclusion for Scaled Data**: The Logistic Regression model produced a 100% percision,
# 67% recall, and 80% F1-score for all instances of '1', and a 83% percision, 100% recall, and 91% F1-score for
# all instances of '0' of the target variable. This means that the model appears to identify Regions with a μ Age
# Maternal Mortality that *is* or *is not* above the μ Age Maternal Mortality in Mexico with marginal to somewhat
# reliable accuracy.

# Compute the confusion_matrix to evaluate the accuracy of a classification
conf_matrix = confusion_matrix(y_test, predict_scaled)

# Plot confusion_matrix
plt.figure(figsize = (10,7))
sns.heatmap(conf_matrix, annot=True)

# Compute predicted probabilities: y_pred_prob
y_pred_prob_scaled = clf_log_scaled.predict_proba(X_test)[:,1]

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(clf_log_scaled, X, y, scoring='roc_auc', cv=5)

# Visualize Logistic Regression Model Accuracy for the Scaled Data with ROC curve
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_scaled)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
