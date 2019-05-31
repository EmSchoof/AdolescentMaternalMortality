Springboard Capstone 1: Adolescent Maternal Mortality in Mexican States

In early 2019, I came across the the website Data Science for Good, where a team of data scientists worked on a project known as 'Making our moms proud: Reducing Maternal Mortality in Mexico'. The purpose of this project was assess why Mexico's Maternal Mortality Ratio (MMR, calculated by the WHO as # of deaths during pregnancy or within 42 days after birth , per 100,000 live birth) "has stagnated (over the past 10 years)despite additional efforts from the government to further bring it down."

As a woman in child-bearing years, it is both exhilerating and troubling to learn about the successes and failures in the advancements of women health, especially when it comes to maternal mortality. My current residence in Southern California resident combined with my growing interest in becoming involved in Mexico-United States of America global relations, I decided to use the 'Making our moms proud' dataset as a learning tool for my first capstone project at Springboard.

The scripts provided here calculate the risk probability of an adolescent maternal mortality by State in Mexico based on some of the top features contributing to maternal mortality. While this is a multi-dimensional issue, for the sake of this study, the following 8 factors were used to predict the likelihood of adolescent maternal mortality by region: region population, region GDP, local poverty level, level of education, and access to medical assistance in order to help direct government funds to areas where it would be most beneficial. 

Use
maternal_mortality.ipynb cleans the source data and assesses mean maternal mortality age and mean adolescent maternal mortality age by State in Mexico.
merging_dataframes.ipynb merges averaged information on State instances of maternal mortality with State enconomic factors (GDP and Population Size).
machine_learning.ipynb produces the linear regression model assessing likelihood of adolescent maternal mortality within a State of Mexico.

Dependencies
The specific Python files written by this code assume you have the following tools added to your module directories:

# General
pandas, pandas DataFrame
numpy
seaborn

# Visualization
matplotlib.pyplot
pylab
statsmodels.distributions.empirical_distribution

# Statistics
statistics
scipy.stats

# Machine Learning modules
DecisionTreeClassifier
RandomForestClassifier
train_test_split
LogisticRegression

# Hyperparameters
GridSearchCV
roc_auc_score
cross_val_score

# ROC Curve
roc_curve

# Scale Data
StandardScaler
