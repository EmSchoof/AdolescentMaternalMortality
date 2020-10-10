# Import the relevant python libraries
import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from numpy import histogram
import seaborn as sns
import scipy.stats as stats
from statsmodels.distributions.empirical_distribution import ECDF

# Load datasets
get_ipython().run_line_magic('store', '-r  materna')
get_ipython().run_line_magic('store', '-r  res_dataset')

### Question 1: What is the Mean Age of Maternal Mortality within the dataset? How does  compare to the
# Actual Mean Age of Maternal Mortality in Mexico?

# Create variable for maternal death
age_mortality = materna['Age at Death']

# Determine sample size for maternal death 
sample_size = len(age_mortality)

##### Plot Sample Age of Maternal Death Distribution

# Create a figure with two plots
fig, (boxplot, histogram) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Add boxplot for maternal death
sns.boxplot(age_mortality, ax=boxplot)

# Remove x-axis label from boxplot
boxplot.set(xlabel='')

# Add histogram and normal curve for maternal death
fit = stats.norm.pdf(age_mortality, np.mean(age_mortality), np.std(age_mortality))
pl.plot(age_mortality, fit, '-o')
pl.hist(age_mortality, density=True, alpha=0.5, bins=20)

# Label axis 
pl.xlabel('Age of Maternal Mortality')
pl.ylabel('Probability Density Function')
pl.title('Age Distribution Associated with the Incidence of Maternal Mortality in Mexico')

# Show plot and add print mean and std sample information
plt.show()
'The sample(n=' + str(sample_size) + ') population mean age of maternal death is ' + str(
    round(np.mean(age_mortality), 2)) + ' years old with a standard deviation of ' + str(
    round(np.std(age_mortality), 2)) + '.'


# *Preliminary* **Conclusion**: The distribution appears to be generally normally distributed based off of
# the histogram of the maternal age at time of death. Since binning bias can occur, the Cumulative Distribution
# Function (CDF) needs to be analyzed. Based off of the Central Limit Theorem (CLT), the sampling distribution of
# the sample means approaches a normal distribution as the sample size ( n ) gets larger - regardless of what the
# shape of the population distribution. Under this theorem,  n>30  is considered a large sample size. Since the
# current database sample size  n  = 16636, CLT can be assumed.

# Create an Empirical and Theoretical Cumulative Distribution Function (CDF)
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y


# Seed a Random Number Generator and Calculate Theoretical Dataset with Normal Distribution
np.random.seed(15)

# Compute the theoretical CDF 
cdf_mean = np.mean(age_mortality)
cdf_std = np.std(age_mortality)

# Simulate a random sample with the same distribution and size of 10,000
cdf_samples = np.random.normal(cdf_mean, cdf_std, size=10000)
c
# Apply the ecdf() function to Empirical and Theoretical Data to Calculate the CDFs

# Compute the CDFs
x_death, y_death = ecdf(age_mortality)
x_norm, y_norm = ecdf(cdf_samples)

# Plot the Theoretical (normalized) CDF with the Empirical (sample) CDF
fig = plt.plot(x_death, y_death, marker='.', linestyle='none', alpha=0.5)
fig = plt.plot(x_norm, y_norm, marker='.', linestyle='none', alpha=0.5)

# Label figure
fig = plt.xlabel('Age of Maternal Death')
fig = plt.ylabel('CDF')
fig = plt.legend(('Sample Population', 'Expected Norm'))
fig = plt.title('Distribution of Maternal-Associated Deaths in Mexico')

# Save plots
plt.show()

# **Conclusion**: Since the normalized data and the sample population's empirical data follow along the same line,
# it can be assumed that the population sample is normally distributed.Therefore, based off of both the Central
# Limit Theorem (CLT) and the Empirical Cumulative Distribution Function (ECDF), the sample population of age of
# maternal death is normally distributed.

# Question 2: What is the Average Age of Maternal Death within each Region of Mexico in the dataset? How does each
# Region Mean compare to the Sample Population mean (28.35)?

# Create a bar graph to show distribution of incidences of maternal death by region
fig, ax = plt.subplots(figsize=(16, 4))
plt.xticks(rotation='vertical')
plt.grid(True)
fig.subplots_adjust(bottom=0.2)
sns.countplot(materna['Residence Name'])
pl.title('Incidence of Maternal Mortality in Each Providence of Mexico')

# *Preliminary Observation*: It appears that certain regions have the incidence of maternal death more prevenant
# than others. Further analysis needs to be performed in order to assess the cause of these differences. For now,
# it's important to understand that each region has a different population size. Are these populations
# distributed normally?

# Create a boxplot to show the distribution of each region compared to its mean
fig, ax = plt.subplots(figsize=(16, 8))
plt.xticks(rotation='vertical')
fig.subplots_adjust(bottom=0.2)
sns.boxplot(x=materna['Residence Name'], y=materna['Age at Death'], data=materna)
pl.title('Age Distribution of Maternal Mortality within Each Providence of Mexico')

# *Preliminary Observation*: As noted above, some Mexican States appear to experience varying instances and age
# distributions of maternal death. Could this be due to the size of the region or possibly the economic status
# of the region?

# Create a bar graph to show mean age maternal death by region
fig, ax = plt.subplots(figsize=(16, 4))
plt.xticks(rotation='vertical')
plt.grid(True)
plt.scatter(res_dataset['Region'], res_dataset['μ Age Maternal Mortality'])
pl.title('Regions Compared to Mexico Mean Age Maternal Mortality')

# ***Further Investigation Needed***: The difference between the means of each region within Mexico needs to be
# analyzed to evaluate if theres differences are statistically significant. This can be accomplished by running
# an ANOVA analysis.

### Question 3: Since the Differences of Means Maternal is Statistically Significant, how do the Region Mean Ages
# Compare to the Mexico's National Mean Age Maternal Mortality?

# Create a bar graph to show mean age maternal death by region
fig, ax = plt.subplots(figsize=(16, 4))
plt.grid(True)
fig.subplots_adjust(bottom=0.2)
sns.countplot(res_dataset['Above(0) or Below(1) Average'])
pl.title('Mean Age Maternal Mortality in Each Providence of Mexico')

# *Preliminary Observation*: It appears that more than half of Mexico's Regions (18 of 32) have a lower mean age of
# maternal mortality than the remaining 14. What are the major differences between these regions?

# List Regions with a mean maternal mortality lower than population mean
len(res_dataset[res_dataset['Above(0) or Below(1) Average'] == 1])

# ## Question 5: How does the average level of Education Completed by individuals who suffered maternal mortality
# change by Region? What about local community size and access to medical care?
# - The World Health Organization (WHO) states that better educated women tend to be healthier, participate more
# in the formal labor market, earn higher incomes, have fewer children, marry at a later age, and enable better
# health care and education for their children, should they choose to become mothers. All these factors combined
# can help lift households, communities, and nations out of poverty.

# Select features of interest
data = res_dataset[['μ Age Maternal Mortality',
                    'μ Region Education Level',
                    'μ Region Local Community Size',
                    'μ Presence(0)/Not(1) of Med Assist']]

# Plot sns pairplot
sns.pairplot(data)

#### Assess correlation of selected variables
feature_corr = data.corr()

# Observations: It appears that the strongest correlations are between μ Region Education Level and μ Region
# Local Community Size (+0.668567), but all have weak to no strong correlation to μ Age Maternal Mortality on
# their own. However, how can these three socioeconomic factors help predict if a Region's μ Age Maternal Mortality
# is above or below the national mean of ~28yrs in Mexico?
