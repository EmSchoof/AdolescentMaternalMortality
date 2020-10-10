# Inferential Statistics: Bootstrap Simulation & ANOVA Analysis

# Import the relevant python libraries for the analysis
import math
import numpy as np
import pandas as pd
import pylab as pl
import random
import seaborn as sns
import scipy.stats as stats
import statistics


# Load datasets
# get_ipython().run_line_magic('store', '-r  materna')
# get_ipython().run_line_magic('store', '-r res_dataset')
# get_ipython().run_line_magic('store', '-r age_by_state')
# get_ipython().run_line_magic('store', '-r state_pop')


# Bootstrap Simulation: Statistical Testing of Data Mean

# $H$o: The mean age maternal mortality of women in Mexico is equal to mean age maternal mortality within the dataset
# ($28.35 yoa$). <br>  Empirical Mean ($μ$) − Population Mean ($μ$) = 0

# $H$a: The mean age maternal mortality of women in Mexico is *not* equal to mean age maternal mortality within
# the dataset  ($28.35 yoa$). <br> Empirical Mean ($μ$) − Population Mean ($μ$) ≠ 0

# Significance Level: *95%* Confidence. <br> $α$ = 0.05

# Create a boostrap replicate function with another function to repeat the bootstrap replication 'x' number of times

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


# Calculate the size, mean, and standard deviation of *materna*[ 'Age at Death' ]

# Define a variable for the materna['Age at Death'] Series
age_mortality = materna['Age at Death']

# Create variables for sample statistical information
materna_age_std = age_mortality.std()
materna_sample_size = len(age_mortality)
materna_age_var = np.var(age_mortality)
mean_age = materna['Age at Death'].mean()

# Create an array of the sample mean that is equal to the boostrap array length
materna_mean_arr = np.full(10000, mean_age)

# Create 10000 bootstrap replicates of the mean and take the mean of the returned array
boot_tenthousand = draw_bs_reps(age_mortality, np.mean, size=10000)

# Calculate the difference between the bootrap and sample means with p-value and standard deviation
# Compute p-value
p_val = np.sum(boot_tenthousand >= materna_mean_arr) / len(boot_tenthousand)

# Calculate the standard margin of error for a 95% confidence interval
conf_int_low = mean_age - (1.98 * (materna_age_std / math.sqrt(materna_sample_size)))
conf_int_high = mean_age + (1.98 * (materna_age_std / math.sqrt(materna_sample_size)))

# **Conclusion from the Bootstrap Hypothesis Test:** The resulting population mean of maternal death approximation
# based on 10,000 bootstrap replicate samples was *28.3564 years of age (yoa)*, which is close to the sample mean of
# *28.3568 yoa* old from the dataset. Additionally, the bootstrap population mean is within the 95% Confidence
# Interval, *28.2443 to 28.4692 yoa* with a p-value of 0.4959, which is greater than α = 0.05. Therefore,
# the null hypothesis that the mean age of death of maternal women in Mexico is equal to the mean age of death
# presented in the dataset can be accepted. **$Ho$ is accepted**.

# ANOVA Analysis

# A **One-Way Analysis of Variance (ANOVA)** test compares the means of two or more groups to determine if at least one
# group mean is statistically different from the others. These assumptions that must be satisfied in order for the
# associated p-value to be valid:
# 1. The samples are independent.
# 2. Each sample is from a normally distributed population.
# 3. The population standard deviations of the groups are all equal. (homoscedasticity)

# #### 1. Samples must be Random / Independent
# 10% Rule: If sample size (n) for each Mexican State is less than 10% of the total population within that State, than
# each sample selection can be treated as an independent event

# Verify both age_by_state and state_pop both contain all 32 Mexican States/Regions 
len(list(age_by_state.keys())), len(state_pop)

# Modify state_pop to contain the extact same string value for State
state_pop['State'] = age_by_state.keys()

### OVER ESTIMATION - If Women are only 30% of Population ###
for state in age_by_state:
    age_arr = age_by_state[state]  # select age arr
    age_length = len(age_arr)  # calculate State age sample size
    state_row = state_pop[state_pop['State'] == state]  # select State row in state_pop

    # Calculate 10% state_pop State populations in 2010 and 2015
    ten_percent_2010 = round(float(state_row['Population 2010']) ** 0.1, 2)
    ten_percent_2015 = round(float(state_row['Population 2015']) ** 0.1, 2)

    # Calculate 10% of 30% of state_pop State populations in 2010 and 2015
    state_10_30 = round(float(state_row['Population 2010']) ** 0.3, 2)
    ten_percent_10_30 = round(state_10_30 ** 0.1, 2)
    state_15_30 = round(float(state_row['Population 2015']) ** 0.3, 2)
    ten_percent_15_30 = round(state_10_30 ** 0.1, 2)

    # Set condition: Compare age_length to 10% and 30% State populations in 2010 and 2015
    if age_length > ten_percent_2010 and age_length > ten_percent_2015:
        print('Accept Independence: ', state)
    else:
        print('REJECT: ', state)

    if age_length > ten_percent_10_30 and ten_percent_10_30:
        print('Accept Independence - Over-Estimation: ', state)
    else:
        print('REJECT - Over-Estimation: ', state)

# **Independence Conclusion**: Since the dataset for the incidence of maternal mortality within each Mexican
# States/Region is less than 10% of the recorded populations of each State in both 2010 and 2015,
# each data point can be treated as an independent variable. **Independence is Accepted**

# 2. Samples must be Normally Distributed
# Within the SciPy module of python 3, there is a normalcy function that tests the null hypothesis that a sample comes
# from a normal distribution. It is based on **D’Agostino** and **Pearson’s test** that combines skew and kurtosis to
# test of normality. This function be used to further determine if the distribution of each Province sample population
# is normally distributed.

# Create a variable to hold list of Regions with normally-distributed sample sizes
norm_distr_regions = []

# Create a variable to hold list of Regions without normally-distributed sample sizes
not_norm_distr_regions = []

# In[13]:


# Determine if each Province has a normally distributed sample population of ages
for region in age_by_state:
    """Determine if Region Age Distribution is Normal"""

    region_name = str(region)
    arr = age_by_state[region_name]

    if len(arr) > 8:  # skewtest (k2): not valid with less than 8 samples
        k2, p = stats.normaltest(arr)
        alpha = 0.05  # 95% confidence
        print("p = {:g}".format(p))
        print("n = " + str(len(arr)))

        if p < alpha:  # if norm
            print(str(region) + " IS normally distributed.")
            norm_distr_regions.append(region_name)  # add region to norm list
        else:
            print(str(region) + " *IS NOT* normally distributed.")
            not_norm_distr_regions.append(region_name)  # add region to norm list
    else:
        print(str(region) + " *sample size is too small*")
        not_norm_distr_regions.append(region_name)  # add region to non-norm list of regions

# In[14]:


print('Not Normally Distributed: ', list(np.unique(not_norm_distr_regions)))

# **Normalcy Conclusion**: After assessing the distribution of age of maternal death within each Province of Mexico,
# **all Province sample populations are considered to be normally distributed** *aside from* Colima
# (p-value: 2.13913e-21, n=1082) Quintana Roo (p-value: 0.00022599, n=375), which were found to *not be normally
# distributed*.
# ​
# However, under the Central Limit Theorem (CLT), the sampling distribution of the sample means approaches a normal
# distribution as the sample size ( n ) gets larger - regardless of what the shape of the population distribution.
# Under this theorem,  n>30  is considered a large sample size. *Since the current database sample size (n) of Colima
# and Quintana Roo of 1082 and 375, respectively, justify CLT being assumed.*

# #### Bartlett’s Test for Homogeneity of Variance
# $H$o: All region age of maternal mortality populations have equal variance. <br>  $v$1 = $v$2 = $v$3 = .... = $v$32
# 
# $H$a: There is at least one region age of maternal mortality population variance is statistically different from the
# rest. <br> $v$1 ≠ $v$2 = .... = $v$32
# 
# Significance Level: *95%* Confidence. <br> $α$ = 0.05

# In[15]:


# Calculate the age variance per region - This section has test trials per entry to help formulate an
# iteration function

# Aquascalientes and Baja California
print(stats.bartlett(age_by_state['Aguascalientes'], age_by_state['Baja California']))

# Aquascalientes and Baja California Sur
print(stats.bartlett(age_by_state['Aguascalientes'], age_by_state['Baja California Sur']))

# *Preliminary* **Conclusion for Homogeneity of Variance**: Even before calculating all Bartlett results between
# region populations, it is evident that not all regions have the same variance. Therefore, **the null hypothesis
# is rejected: there is at least one region age of maternal mortality with a population variance statistically
# different from the rest**. Luckily, the proof for ANOVA test is robust, so slight variations from its proof
# criteria are  OK (source: https://faculty.elgin.edu/dkernler/statistics/ch13/13-1.html). As a good rule of thumb,
# *as long as the largest variance is no more than double the smallest, we can assume ANOVA's requirement for
# Homogeneity of Variance is satisfied.*

# Evaluate the differences between the largest and smallest Region variances in Mexico.

# Create variables for minimum and maximum variation values in res_dataset
max_variance = res_dataset['μ Age Variance'].max()
min_variance = res_dataset['μ Age Variance'].min()

# Check if largest variance is more than double the smallest
if (2 * min_variance) >= max_variance:
    print('Accept ANOVA: The max variance is less than double the min variance.')
else:
    print('Reject ANOVA: The max variance is more than double the min variance.')

# **Final Conclusion for Homogeneity of Variance**: Since the maximum variance within the dataset (61.68) is
# less than double the minimum variance within the dataset (40.12, whichs doubles to 80.24), we can assume
# ANOVA's requirement for Homogeneity of Variance is satisfied.

# ### Calculate One-Way Analysis of Variance
# 
# $H$o: All mean age of death of maternal women within all Province of Mexico are statistically similar.
# $μ$1 = $μ$2 = $μ$3 = .... = $μ$32
# 
# $H$a: There is at least one mean age of death of maternal women within a Province of Mexico that is statistically
# different from the rest. <br> $μ$1 ≠ $μ$2 = .... = $μ$32
# 
# Significance Level: *95%* Confidence. <br> $α$ = 0.05

# Define the number of conditions (k) based on Region/State
k = len(pd.unique(materna['Residence Name']))

# Calculate the conditions times data points (N)
N = len(materna.values)

# Participants in each condition
n = materna.groupby('Residence Name').size()[0]

# Create a dict variable for key:value pairs of state:age_array
grps = pd.unique(materna['Residence Name'].values)
state_mean_ages = {grp: materna['Age at Death'][materna['Residence Name'] == grp] for grp in grps}

# Calculate the ANOVA F- value and p-value using stats module
F, p = stats.f_oneway(state_mean_ages['Aguascalientes'],
                      state_mean_ages['Baja California'],
                      state_mean_ages['Baja California Sur'],
                      state_mean_ages['Campeche'],
                      state_mean_ages['Chiapas'],
                      state_mean_ages['Chihuahua'],
                      state_mean_ages['Coahuila de Zaragoza'],
                      state_mean_ages['Colima'],
                      state_mean_ages['Distrito Federal'],
                      state_mean_ages['Durango'],
                      state_mean_ages['Guanajuato'],
                      state_mean_ages['Guerrero'],
                      state_mean_ages['Hidalgo'],
                      state_mean_ages['Jalisco'],
                      state_mean_ages['Michoacán de Ocampo'],
                      state_mean_ages['Morelos'],
                      state_mean_ages['México'],
                      state_mean_ages['Nayarit'],
                      state_mean_ages['Nuevo León'],
                      state_mean_ages['Oaxaca'],
                      state_mean_ages['Puebla'],
                      state_mean_ages['Querétaro Arteaga'],
                      state_mean_ages['Quintana Roo'],
                      state_mean_ages['San Luis Potosí'],
                      state_mean_ages['Sinaloa'],
                      state_mean_ages['Sonora'],
                      state_mean_ages['Tabasco'],
                      state_mean_ages['Tamaulipas'],
                      state_mean_ages['Tlaxcala'],
                      state_mean_ages['Veracruz de Ignacio de la Llave'],
                      state_mean_ages['Yucatán'],
                      state_mean_ages['Zacatecas'])

# **ANOVA Conclusion**: ANOVA was performed using a confidence level of 95%. The resulting p-value was 5.79x10-13,
# which is substantially smaller than α = 0.05. Thus, **the null hypothesis is rejected: the differences in mean
# age of maternal mortality across the regions in Mexico are statistically different**. Since adolescent maternal
# mortality is one of the key factors that lowers the mean age per region, it would be beneficial to calculate the
# mean age adolescent maternal mortality in preparation for the machine learning assessment of mean maternal age
# of mortality.
