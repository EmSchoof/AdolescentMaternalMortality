# Import the relevant python libraries for the analysis
import pandas as pd
from pandas import DataFrame
import numpy as np
import statistics

# Load the dataset
mortalitad_materna = pd.read_csv('../data/mortalidad_materna.csv')

#### Create a **materna** subdataset *in English* from the **maternalidad_materna** dataset *in Spanish*
# with information including:
# 
# - Residence Area
# - Local Community Size
# - Total Education Completed 
# - Last Recorded Age
# - Reason for Mortality
# - Medical Assistance

#### Step 1: Explore the Data and Create Clean Sub-Dataframe for Analysis

# Combine patient birthdate information into one column
birth = DataFrame(mortalitad_materna, columns=['Año de nacimiento', 'Mes de nacimiento', 'Día de nacimiento'])
birth = mortalitad_materna['Año de nacimiento'].map(str) + '-' + mortalitad_materna['Mes de nacimiento'].map(
    str) + '-' + mortalitad_materna['Mes de nacimiento'].map(str)

# Combine patient date of death information into one column
death = DataFrame(mortalitad_materna, columns=['Año de la defunción', 'Mes de la defunción', 'Día de la defunción'])
death = mortalitad_materna['Año de la defunción'].map(str) + '-' + mortalitad_materna['Mes de la defunción'].map(
    str) + '-' + mortalitad_materna['Mes de la defunción'].map(str)

# #### Create variables for important location, economic, and educational factors associated with the
# instance of maternal mortality:
# - Residence Area
# - Local Community Size
# - Total Education Completed 
# - Last Recorded Age
# - Reason for Mortality
# - Medical Assistance

# Create variable to store:

# residence information
residence_code = mortalitad_materna['Entidad de residencia']
residence_name = mortalitad_materna['Descripción de entidad de residencia']

# local community info
local_size = mortalitad_materna['Descripción del tamaño de localidad']
local_size_code = mortalitad_materna['Tamaño de localidad']

# educational level
edu_reached_code = mortalitad_materna['Escolaridad']
edu_reached = mortalitad_materna['Descripción de la escolaridad']

# age fulfilled by patient
last_age = mortalitad_materna['Edad cumplida']

# mortality reason
mortality_reason = mortalitad_materna['Razón de mortalidad materna']

# medical assistance
medical_received = mortalitad_materna['Descripción de la asistencia médica']

# Create a sub-dataframe to hold all date- information 
materna = pd.concat([birth,
                     death,
                     residence_code,
                     residence_name,
                     local_size,
                     local_size_code,
                     edu_reached_code,
                     edu_reached,
                     last_age,
                     mortality_reason,
                     medical_received], axis=1)
materna.columns = ['Date of Birth',
                   'Date of Mortality',
                   'Residence Code',
                   'Residence Name',
                   'Local Community Size',
                   'Local Size Code',
                   'Education Code',
                   'Education Completed',
                   'Age at Death',
                   'Reason for Mortality',
                   'Medical Assistance Received']

#### Data Wrangling:

##### Part 1: Check for null or errors within materna


# Order dataframe to list in ascending order of approx. age at death
materna = materna.sort_values(by=['Age at Death'], ascending=True)
materna = materna.reset_index(drop=True)

##### Check if all values make sense
# - Case 1: It is biologically impossible to reach 998 years of age. These data entries appear to all have '0-0-0'
# in the 'Date of Birth' column and should therefore be removed from the sub-dataset, dates.
# - Case 2: What do the '0' and '1' entires for 'Reason for Mortality' mean? If one of these keys is not associated
# with maternal death, then those entries should also be removed from the sub-dataset, dates.

# Remove rows with NaN / '0-0-0' values in Date of Birth
materna = materna[materna['Date of Birth'] != '0-0-0']

# **Case 1 Conclusion**: It appears that removing rows with '0-0-0' in the 'Date of Birth' column did remove all
# entries outside of the biological scope of maximum age at death. However, it is unlikely that a women who reached
# the age of 81 died due to maternal reasons since this age is outside the childe-bearing years. The analysis of
# the '0' and '1' in the 'Reason for Mortality' column may shed more light since all younger ages seem to be
# associated with '1' while older ages are associated with '0'.

# Create a variable for the description of Reason for Mortality Description
mortality_description = mortalitad_materna['Descripción de la razón de mortalidad materna']

# Create a sub-dataframe to show interaction of Reason for Mortality Code and Description
mortality = pd.concat([mortality_reason, mortality_description], axis=1)
mortality.columns = ['Reason Mortality Code', 'Reason Mortality Description']

# **Case 2 Conclusion**: Since '1' refers to recorded maternal-deaths and '0' refers to recorded deaths that are
# *not* associated with maternity, all rows containing '0' in the 'Reason for Mortality' column should be removed.

# Remove rows with 0 values in Reason for Mortality
materna = materna[materna['Reason for Mortality'] != 0]

# #### Part 2: Translating Important Columns in **materna** from Spanish *using translation_english.txt*
# - Important columns that need translation include: 
#     - *Education Completed*
#         - Translate Spanish Descriptions into Integer Values that Are Comparable
#     - *Local Community Size*
#         - Translate Spanish Descriptions into Integer Values that Are Comparable
#     - *Medical Assistance Received*
#         - Translate Medical Assistance Received into a Binary Column

# **Education Completed**
# - 9
#     - 'POSGRADO' = Post-Graduate Education
# - 8
#     - 'PROFESIONAL' = Professional School
# - 7
#     - 'BACHILLERATO O PREPARATORIA COMPLETA' = High School (grades 10-12) complete
# - 6
#     - 'BACHILLERATO O PREPARATORIA INCOMPLETA' = High School (grades 10-12) incomplete
# - 5 
#     - 'PRIMARIA COMPLETA' = Elementary School (grades 1-6) complete 
# - 4 
#     - 'PRIMARIA INCOMPLETA' = Elementary School (grades 1-6) incomplete
# - 3 
#     - 'SECUNDARIA COMPLETA' = Junior High (grades 7-9) complete
# - 2
#     - 'SECUNDARIA INCOMPLETA' = Junior High (grades 7-9) incomplete
# - 1
#    - 'PREESCOLAR' = Preschool complete
# -  0 
#     - Combine the following entries:'SE IGNORA' = It was 'ignored' /
#                                     'NO ESPECIFICADO' = Not specified / 'NINGUNA' = NONE


# Create a sub-dataframe to show interaction of Education Code and Education Completed
education = materna[['Education Code', 'Education Completed']].sort_values(by='Education Code')
education = education.drop_duplicates()

# Overwriting column with replaced value of Education

# SE IGNORA / NINGUNA / NO ESPECIFICADO
materna["Education Completed"] = materna["Education Completed"].replace(['SE IGNORA', 'NINGUNA', 'NO ESPECIFICADO'], 0)

# PREESCOLAR
materna["Education Completed"] = materna["Education Completed"].replace('PREESCOLAR', 1)

# PRIMARIA
# INCOMPLETA
materna["Education Completed"] = materna["Education Completed"].replace('PRIMARIA INCOMPLETA', 2)
# COMPLETA
materna["Education Completed"] = materna["Education Completed"].replace('PRIMARIA COMPLETA', 3)

# SECUNDARIA
# INCOMPLETA
materna["Education Completed"] = materna["Education Completed"].replace('SECUNDARIA INCOMPLETA', 4)
# COMPLETA
materna["Education Completed"] = materna["Education Completed"].replace('SECUNDARIA COMPLETA', 5)

# BACHILLERATO O PREPARATORIA
# INCOMPLETA
materna["Education Completed"] = materna["Education Completed"].replace('BACHILLERATO O PREPARATORIA INCOMPLETA', 6)
# COMPLETA
materna["Education Completed"] = materna["Education Completed"].replace('BACHILLERATO O PREPARATORIA COMPLETA', 7)

# PROFESIONAL
materna["Education Completed"] = materna["Education Completed"].replace('PROFESIONAL', 8)

# POSGRADO
materna["Education Completed"] = materna["Education Completed"].replace('POSGRADO', 9)

# **Local Community Size**

# Create a sub-dataframe to show interaction of Education Code and Education Completed
local_community = materna[['Local Size Code', 'Local Community Size']].sort_values(by='Local Size Code')
local_community = local_community.drop_duplicates()
print(len(local_community))
local_community

# *Observations*: Local Community Size appears to have corresponding codes ordered from least to greatest
# community size already. Thereofore the 'Local Size Code' values can remain while the 'Local Community Size'
# column can be dropped.

# #### Medical Assistance Received
# - 0: WITH Medical Assistance
# - 1: Unspecified/WITHOUT Medical Assistance


# Create a list item to hold comparison response
binary_medassist = []

# Create an iteration function to compare region mean to popupation mean
for medassist in materna['Medical Assistance Received']:

    # test for assistance
    if medassist == 'CON ATENCION MEDICA':
        binary_medassist.append(0)
    else:
        binary_medassist.append(1)

# Convert the list to a Series and add as new column
materna['Received(0)/Not(1) Medical Assistance'] = pd.Series(binary_medassist)

# Drop columns that are unnecessary
materna = materna.drop(
    columns=['Date of Birth', 'Date of Mortality', 'Local Community Size', 'Medical Assistance Received',
             'Education Code', 'Reason for Mortality'])

# ### Additional Cleaning of Data: 
# There are 31 states and 1 federal entity in Mexico, so the length of the 'Regions' column should be 32, not 34.
# When translating the list of regions within the sub-dataset, it becomes apparent that not all entries are Provinces
# within Mexico, and need to be removed. Namely:
# - Estados Unidos de Norteamérica - 'United States of America'
# - Otros paises latinoamericanos - 'Other Latin American countries'
# - No especificado - 'Not Specified'
# - Otros paises - 'Other Countries'

# Remove unnecessary rows from region_ages sub-dataset
materna = materna[materna['Residence Name'] != 'Estados Unidos de Norteamérica']
materna = materna[materna['Residence Name'] != 'Otros paises latinoamericanos']
materna = materna[materna['Residence Name'] != 'No especificado']
materna = materna[materna['Residence Name'] != 'Otros paises']

# Store as a global variable that can be uploaded to other Jupyter Notebooks
# get_ipython().run_line_magic('store', 'materna')


# ### Preparation of Data for Machine Learning Analysis
# - Create variables for age distribution by region
# - Create a sub-dataframe for Machine Learning model

# ### Part 1: Create a sample region array variables to hold age distribution per region 

# Test code to create function
aqua = materna[materna['Residence Name'] == 'Aguascalientes']
aqua = aqua['Age at Death']
aqua = np.array(aqua)

mex = materna[materna['Residence Name'] == 'México']
mex = mex['Age at Death']
mex = np.array(mex)


# Since it appears that the sample size of ages of maternal death within the Provinces varies, the total sample
# per Province should be stored in unique age array variables. The process of creating the age array is repeatable,
# so a function should be created then applied to each Province. The array of ages variable can then be stored in a
# dictionary as a value with the associated Province as the key.

# Create a function to group all ages associated with materna death within a Province and store the ages in an array
def age_array(str):
    """Create arrays for all Ages of Maternal Death within a Region"""
    ages = materna[materna['Residence Name'] == str]  # select the region 'str' from the 'Region' column
    ages = ages['Age at Death']  # select the ages within the region
    ages = np.array(ages)  # store the ages in an array
    return ages  # return the unique array


# Create a variable for 'Region' names using np.unique()
list_regions = np.unique(materna['Residence Name'])

# Create an empty dictionary to hold the {Region : region_age_array} key pairs
age_by_state = {}

# Use the age_array function with iteration over residence to create the {Region : region_age_array} key pairs
for region in list_regions:
    age_by_state[region] = age_array(region)  # add arrays as values in dictionary with region-key

# Store as a global variable 
# get_ipython().run_line_magic('store', 'age_by_state')


# ### Part 2: Create a sub-dataframe for the Machine Learning Model
# - residence name
# - residence code (index)
# - region mean
# - region sample size (n)
# - region variance
# - binary target
# - mean educational level
# - mean local community size
# - mean presence of medical care

# Var for residence name 
residence_uniq = np.unique(materna['Residence Name'])

# Var for residence code
residence_code = np.unique(materna['Residence Code'])

# Create the sub-dateframe for region and region code
res_dataset = pd.DataFrame(residence_uniq, index=residence_code)
res_dataset = res_dataset.rename(columns={0: 'Region'})

# #### Calculate the Mean Age per Region
# Test Code
mean_death_list_trial = []

aguas = materna[materna['Residence Code'] == 1]
aguas = aguas[['Residence Code', 'Age at Death']]
aguas_mean = aguas['Age at Death'].mean()
aguas_mean = '{0:0.2f}'.format(aguas_mean)
mean_death_list_trial.append(aguas_mean)

baja = materna[materna['Residence Code'] == 2]
baja = baja[['Residence Code', 'Age at Death']]
baja_mean = baja['Age at Death'].mean()
baja_mean = '{0:0.2f}'.format(baja_mean)
mean_death_list_trial.append(baja_mean)

# Create an empty list to store region sample size and mean age of maternal death
region_mean = []
region_n = []

# Calculate the mean age of maternal death per region
for i in materna['Residence Code'].sort_values().unique():
    """Calculate Length of Age Array and Mean Age per Region"""

    sub_df = materna[materna['Residence Code'] == (i - 1)]  # select one region
    n = len(sub_df['Age at Death'])  # calculate sample length
    mean = sub_df['Age at Death'].mean()  # calculate mean of region
    region_n.append(round(n, 2))  # append n to list
    region_mean.append(round(mean, 2))  # append mean to list

# Convert the list to a Series and add as new column
res_dataset['μ Age Maternal Mortality'] = pd.Series(region_mean)
res_dataset['Region (n)'] = pd.Series(region_n)

# *Preliminary Observation*: Since the region 'Zacatecas' has a NaN value for mean age of maternal death, the
# contents of 'Zacatecas' need to be adjusted from NaN to the actual mean of the data for the res_dataset.
# *With a quick reference, the initial values in the res_dataset match the individually calculated mean for both
# Aguascalientes and Baja California, so we know the NaN is not due to shifted values.*

# #### Clean Data by Replacing NaN/Null values with the Correct Data

# Calculate the mean Age of Death for region 'Zacatecas'
zaca = materna[materna['Residence Code'] == 32]
zaca = zaca['Age at Death']

# Calculate sample size
zaca_n = len(zaca)

# Calculate mean
zaca_mean = zaca.mean()
zaca_mean = round(mean, 2)

# Change contents of res_dataset NaN to calculated mean
res_dataset['μ Age Maternal Mortality'] = res_dataset['μ Age Maternal Mortality'].replace(np.nan, zaca_mean)
res_dataset['Region (n)'] = res_dataset['Region (n)'].replace(np.nan, zaca_n)

# #### Calculate the Age Variance by Region

# Test code
aguas = materna[materna['Residence Code'] == 1]
aguas = aguas[['Residence Code', 'Age at Death']]
aguas_var = statistics.pvariance(aguas['Age at Death'])

baja = materna[materna['Residence Code'] == 2]
baja = baja[['Residence Code', 'Age at Death']]
baja_var = statistics.pvariance(baja['Age at Death'])

# Create an empty list to store age of maternal death variance per region
region_var = []

for i in materna['Residence Code'].sort_values().unique():
    """Calculate Age Standard Deviation and Age Variance per Region"""

    sub_df = materna[materna['Residence Code'] == i]
    age = list(sub_df['Age at Death'])
    var = statistics.pvariance(age)  # calculate age variance of region pop

    for region in sub_df['Residence Name'].unique():  # prevent repeat entries in lists
        region_var.append(round(var, 2))  # append var to region list

# Convert the list to a Series and add as new column
res_dataset['μ Age Variance'] = pd.Series(region_var, index=np.arange(1, 33))

# #### Create Binary Columns:
# 
# μ Age 
# - 0 : Region μ Age Maternal Mortality is *greater than or equal to* the population mean
# - 1 : Region μ Age Maternal Mortality is *less than* the population mean
res_mean_age = res_dataset['μ Age Maternal Mortality'].mean()

# Create a dictionary item to hold comparison response
binary_mean = []

# Compare region mean to population mean
for mean in res_dataset['μ Age Maternal Mortality']:
    if mean >= res_mean_age:
        binary_mean.append(0)
    else:
        binary_mean.append(1)

# Convert the list to a Series and add as new column
res_dataset['Above(0) or Below(1) Average'] = pd.Series(binary_mean, index=np.arange(1, 33))

# #### Calculate Mean Educational Level per Region

# Create an empty list to store region sample size and mean age of maternal death
region_education = []
edu_dict = {}

# Create an iteration function to calculate the mean age of maternal death per region
for i in materna['Residence Code'].sort_values().unique():
    """Calculate Mean Education per Region"""

    sub_df = materna[materna['Residence Code'] == i]
    region = str(sub_df['Residence Name'].unique())
    education = sub_df['Education Completed'].mean()
    mean_edu = round(education, 2)
    region_education.append(mean_edu)
    edu_dict[region] = mean_edu

# Store as a global variable
# get_ipython().run_line_magic('store', 'edu_dict')

# Convert the list to a Series and add as new column
res_dataset['μ Region Education Level'] = pd.Series(region_education, index=np.arange(1, 33))

# #### Calculate Mean Educational Level per Region

# Create an empty list to store region sample size and mean age of maternal death
region_community_size = []
size_dict = {}

# Create an iteration function to calculate the mean age of maternal death per region
for i in materna['Residence Code'].sort_values().unique():
    """Calculate Mean Local Community Size per Region"""

    sub_df = materna[materna['Residence Code'] == i]
    region = str(sub_df['Residence Name'].unique())
    local_community = sub_df['Local Size Code'].mean()
    mean_size = round(local_community, 2)
    region_community_size.append(mean_size)
    size_dict[region] = mean_size

# Store as a global variable
# get_ipython().run_line_magic('store', 'size_dict')


# Convert the list to a Series and add as new column
res_dataset['μ Region Local Community Size'] = pd.Series(region_community_size, index=np.arange(1, 33))

# #### Calculate Mean Presence of Medical Care per Region

# Create an empty list to store region sample size and mean age of maternal death
region_medical = []
medical_dict = {}

# Create an iteration function to calculate the mean age of maternal death per region
for i in materna['Residence Code'].sort_values().unique():
    """Calculate Mean Education per Region"""

    sub_df = materna[materna['Residence Code'] == i]
    region = str(sub_df['Residence Name'].unique())
    med_assist = round(sub_df['Received(0)/Not(1) Medical Assistance'].mean(), 2)
    region_medical.append(med_assist)
    medical_dict[region] = med_assist

# Store as a global variable
# get_ipython().run_line_magic('store', 'medical_dict')


# Convert the list to a Series and add as new column
res_dataset['μ Presence(0)/Not(1) of Med Assist'] = pd.Series(region_medical, index=np.arange(1, 33))

# Store as a global variable 
# get_ipython().run_line_magic('store', 'res_dataset')
