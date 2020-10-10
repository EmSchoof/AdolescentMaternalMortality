### Assessment of Adolescent Maternal Mortality by Region

# ### How do the Region Mean Adolescent Ages Compare to the Mexico's National Mean Adolescent Age Maternal Mortality?  
# - The World Health Organization (WHO) states that an upwards of 13 million *adolescent girls (ages under 20)*
# give birth every year, and complications from those pregnancies and childbirth are a leading cause of death for
# those young mothers.

# Import the relevant python libraries for the analysis
import pandas as pd
from pandas import DataFrame
import pylab as pl
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statistics

# Load datasets
# get_ipython().run_line_magic('store', '-r  materna')
# get_ipython().run_line_magic('store', '-r res_dataset')

# Create an adolescent sub dataframe from materna
adolescent_matern_mortality = materna[materna['Age at Death'] <= 20]

##### Create variables for age distribution by region

# Create a variable for adolescent_ages_maternal_mortality
adolescent_ages = adolescent_matern_mortality['Age at Death']

# Create a variable for adolsecent_sample_size
adolsecent_sample_size = len(adolescent_ages)

#### Visualize Data

# Create a figure with two plots
fig, (boxplot, histogram) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Add boxplot for maternal death
sns.boxplot(adolescent_ages, ax=boxplot)

# Remove x-axis label from boxplot
boxplot.set(xlabel='')

# Add histogram and normal curve for maternal death
fit = stats.norm.pdf(adolescent_ages, np.mean(adolescent_ages), np.std(adolescent_ages))
pl.plot(adolescent_ages, fit, '-o')
pl.hist(adolescent_ages, density=True, alpha=0.5, bins=20)

# Label axis 
pl.xlabel('Adolescent Ages of Maternal Mortality')
pl.ylabel('Probability Density Function')
pl.title('Adolescent Age Distribution of Maternal Mortality in Mexico')

# Show plot and add print mean and std sample information
plt.show()
'The sample(n=' + str(adolsecent_sample_size) + ') population mean age of adolescent maternal mortality is ' + \
str(round(np.mean(adolescent_ages), 2)) + ' years old with a standard deviation of ' + \
str(round(np.std(adolescent_ages), 2)) + '.'

# *Preliminary Observation*: The distribution appears to have a skewed-right distributed based off of the histogram
# and boxplot of instance of adolescent maternal mortality.

# Create a boxplot
adolescent_matern_mortality.boxplot('Age at Death', by='Residence Name', figsize=(12, 8))
plt.xticks(rotation='vertical')

# *Preliminary Observations*: When visualizing the spread of adolescent maternal mortality by region, it appears
# there there are several regions with outliers of very young ages of maternal mortality. Since outliers not
# incorporated in the calculation of the mean, the adolescent age arrays should be added to the res_dataset and
# included in the Linear Regression analysis.

#### Bootstrap Simulation: Compare the Sample Mean to a Statistically-Likely Population Mean
##### Statistical Testing of Data Mean Adolescent Mortality

# $H$o: The mean adolescent age of maternal mortality in Mexico is equal to the mean adolescent age of maternal
# mortality presented in the dataset ($17.98 yoa$). <br>  Empirical Mean ($μ$) − Population Mean ($μ$) = 0

# $H$a: The mean adolescent age of maternal mortality in Mexico is *not* equal to the mean adolescent age of
# maternal mortality presented in the dataset ($17.98 yoa$). <br> Empirical Mean ($μ$) − Population Mean ($μ$) ≠ 0

# Significance Level: *95%* Confidence. <br> $α$ = 0.05

# Create variables for sample statistical information
adolescent_ages_std = adolescent_ages.std()
mean_adolescent_ages = adolescent_ages.mean()

# Create an array of the sample mean that is equal to the boostrap array length
adolescent_ages_arr = np.full(10000, mean_adolescent_ages)


# Bootstrap replicate function for repeatability
def bootstrap_replicate_1d(data, func):
    """Create a bootstrap replicates."""
    boot_sample = np.random.choice(data, size=len(data))  # create bootstrap sample
    return func(boot_sample)  # apply function to bootstrap


# Apply bootstrap replicate function 'n' and return an array
def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""
    boot_rep = np.empty(size)  # initialize array of replicates: bs_replicates
    for i in range(size):  # generate 'n' number of replicates
        boot_rep[i] = bootstrap_replicate_1d(data, func)
    return boot_rep


# Create 10000 bootstrap replicates of the mean and take the mean of the returned array
boot_tenthousand = draw_bs_reps(adolescent_ages, np.mean, size=10000)

# Compute p-value
p_val = np.sum(boot_tenthousand >= adolescent_ages_arr) / len(boot_tenthousand)

# Calculate the standard margin of error for a 95% confidence interval
conf_int_low = mean_adolescent_ages - (1.98 * (adolescent_ages_std / math.sqrt(adolsecent_sample_size)))
conf_int_high = mean_adolescent_ages + (1.98 * (adolescent_ages_std / math.sqrt(adolsecent_sample_size)))

# **Conclusion from the Bootstrap Hypothesis Test:** The resulting population mean of maternal death approximation
# based on 10,000 bootstrap replicate samples was *17.9765 years of age (yoa)*, which is close to the sample mean
# of *17.9767 (yoa)* from the dataset. Additionally, the bootstrap population mean is within the 95% Confidence
# Interval, *17.9139 to 18.0397 (yoa)* with a p-value of 0.4990, which is greater than α = 0.05. Therefore, the
# null hypothesis that the mean age of death of maternal women in Mexico is equal to the mean age of death
# presented in the dataset can be accepted. **$Ho$ is accepted**.

# Create a dict variable for key:value pairs of state:age_array
grps = pd.unique(adolescent_matern_mortality['Residence Name'].values)
state_mean_adolescent_ages = {
    grp: adolescent_matern_mortality['Age at Death'][adolescent_matern_mortality['Residence Name'] == grp] for grp in
    grps}

# Create an empty list to store mean age and sample size of maternal death per region
ado_region_mean = []
ado_region_n = []

# Create an iteration function
for i in adolescent_matern_mortality['Residence Code'].sort_values().unique():
    """Calculate Mean Age Adolescent Maternal Mortality per Region"""
    sub_df = adolescent_matern_mortality[adolescent_matern_mortality['Residence Code'] == i]
    age = sub_df['Age at Death']
    ado_n = len(age)  # sample length
    ado_mean = age.mean()  # calculate mean

    for region in sub_df['Residence Name'].unique():  # prevent repeat entries in lists
        ado_region_mean.append(round(ado_mean, 2))  # append mean to region list
        ado_region_n.append(round(ado_n, 2))  # append ado_n to region list

# Convert the list to a Series and add as new column
res_dataset['μ Age Adolescent Maternal Death'] = pd.Series(ado_region_mean, index=np.arange(1, 33))
res_dataset['Region Ado (n)'] = pd.Series(ado_region_n, index=np.arange(1, 33))

# Store as a global variable 
# get_ipython().run_line_magic('store', 'res_dataset')
