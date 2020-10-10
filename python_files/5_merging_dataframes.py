#!/usr/bin/env python
# coding: utf-8

# ### Create a Combined Database with Mean Age and Variance by Region Maternal Mortality, Mean and Variance of
# Adolescent Age Maternal Mortality by Region, GDP by Region, and Population by Region in Mexico.
# - This requires the combination of res_dataset (with each Region mean and variance for total and adolescent
# maternal mortality), metro_by_region (with Metropolitan Areas organized by Region), and mexico_gdp (with GDP
# values organized by Metropolitan Areas).

# Import the relevant python libraries for the analysis
import pandas as pd
from pandas import DataFrame
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl


# Statistics
import statistics
import scipy.stats as stats
import random
import math
import re
#get_ipython().run_line_magic('matplotlib', 'inline')

# Load and test Metro by Region dataset and set the index if applicable
metro_by_region = pd.read_excel('../data/metro_area_by_state.xlsx')

# Load and test Mexico GDP by Region dataset and set the index if applicable
mexico_gdp = pd.read_csv('../data/mexico_region_gdp_per_capita.csv')

# Step 1: Explore metro_by_region and mexico_gdp Datasets and Create Clean Sub-Dataframes for Analysis

# Remove Irregular Values
# - Case 1: Remove all rows with 'NaN' in the column values.

#Drop NaN values 
metro_by_region = metro_by_region.dropna()

# Organize dataset by State in alphabetical order
metro_by_region = metro_by_region.sort_values(by=['State(s)'],ascending=True)

# Reset index 
metro_by_region = metro_by_region.reset_index(drop=True)

# *Preliminary Observations*: When exploring the data found in mexico_gdp, I found the GDP data for Campeche and Mexico were missing. I assumed that the GDP values did not transfer over from OCED.stat, but upon further review of the OCED.stat database, Campeche and Mexico GDP information were not available. Since GDP is to be used as a potential predictive factor for maternal mortality, the assessment of Campeche, individually, will be weakened without this data. Therefore, although there are values missing, the overall purpose of this dataset is to merge it with other factors potentially involved in maternal mortality, so these rows will remain in the dataset.

# Create a Dataframe to store data types in mexico_gdp
mex_gdp_dtypes = pd.DataFrame(mexico_gdp.dtypes)

#Rename column name to DataType
mex_gdp_dtypes = mex_gdp_dtypes.rename(columns={0:'DataType'})

#Analyze Missing Values
mex_gdp_dtypes['MissingVal'] = mexico_gdp.isnull().sum()

#Identify number of unique values
mex_gdp_dtypes['NumUnique'] = mexico_gdp.nunique()

#Identify the count for each variable
mex_gdp_dtypes['Count']= mexico_gdp.count()

# **Conclusion**: The dataset seems rather clean as there are no missing values aside from those referenced
# above. Now, the list of 'Metropolitan areas' needs to be catagorized into the 32 Provinces within Mexico.

# Convert number objects in Year_2010 and Year_2015 to numbers

# Year_2010
mexico_gdp['Year_2010'] = pd.to_numeric(mexico_gdp['Year_2010'], errors='coerce')

# Year_2015
mexico_gdp['Year_2015'] = pd.to_numeric(mexico_gdp['Year_2015'], errors='coerce')

# Run descriptive statistics of number datatypes
mexico_gdp.describe(include=['number'])

# Reindex by Metropolitan Area
mexico_gdp = mexico_gdp.sort_values(by=['Metropolitan Areas'],ascending=True)

# Reset the Index
mexico_gdp = mexico_gdp.reset_index(drop=True)

# Change the column names to be more clear
mexico_gdp.columns = ['Metropolitan Areas', 'Metro GDP 2010', 'Metro GDP 2015']

# Fix the Metropolitan Names in Preparation for Data Merging

# Create an empty list to store new Metropolitan Areas string
metro_areas = []

# Iterate through Metropolitan Areas to Remove the MEX## from the string and add to metro_areas
for metro in mexico_gdp['Metropolitan Areas']:
    """Remove MEX##: from Metropolitan Area Strings."""
    
    # Remove all numbers from each Metro Area string
    metro_no_numbers = re.sub("\d+", " ", metro)
    
    # Remove all 'MEX : ' from each Metro Area string
    metro_new_string = metro_no_numbers.replace('MEX : ', '')
    metro_new_string = metro_new_string.replace('MEX: ', '')
    
    # Create new Metro Area Column in mexico_gdp with metro_str_modified
    metro_areas.append(metro_new_string)

# Convert the list to a Series and add as new column to res_dataset
mexico_gdp['Metro Areas'] = pd.Series(metro_areas)

# Drop 'Metropolitan Areas' column as it is unnecessary
mexico_gdp = mexico_gdp.drop(columns='Metropolitan Areas')

# Step 2: Merging 3 Dataframes
# - Part 1: Match *metro_by_region[State(s]* dataset with *res_dataset* dataset by the 'Providence' columns
# - Part 2: Match *metro_by_region['Name']* dataset with *res_dataset* dataset by the 'Metropolitan' columns
# - Part 3: Merge *metro_by_region* with *mexico_gdp*
# - Part 4: Merge combined *metro_by_region*/*mexico_gdp* with *res_dataset*

# Part 1: Prepare Data for merging *metro_by_region['State(s)']* with *dates['Residence Name']*

# Assess States/Provinces format in *res_dataset['Region']*

# Open res_dataset as a global variable that can be uploaded to other Jupyter Notebooks
#get_ipython().run_line_magic('store', '-r res_dataset')

# Assess States/Provinces format in *metro_by_region['State(s)']*

# #### Observations:
# As stated when originally cleaning the *res_dataset* in average_age_maternal_mortality Jupyter Notebook, there are 31 states and 1 federal entity in Mexico. Therefore, the length of the metro_by_regions['State(s)'] column is too long at 37 entries. When looking over the list of regions within the dataset, it becomes apparent that there are duplicate entries of States within Mexico, and need to be combined together. Namely:
# - 'Coahuila de Zaragoza' = 'Coahuila de Zaragoza' + 'Coahuila de Zaragoza / Durango'
# - 'Guanajuato' =  'Guanajuato'+ 'Guanajuato / Michoacán de Ocampo'
# - 'Jalisco' =  'Jalisco'+ 'Jalisco / Nayarit'
# - 'Puebla' = 'Puebla' + 'Puebla / Tlaxcala'

# Fix State Names in metro_by_region['States'] dataset
# Replace the duplicate 5 Mexican State names with name matching the dates dataset 

#Coahuila de Zaragoza
metro_by_region = metro_by_region.replace('Coahuila de Zaragoza / Durango', 'Coahuila de Zaragoza')

#Guanajuato
metro_by_region = metro_by_region.replace('Guanajuato / Michoacán de Ocampo','Guanajuato')

#Jalisco
metro_by_region = metro_by_region.replace('Jalisco / Nayarit', 'Jalisco')

#Puebla
metro_by_region = metro_by_region.replace('Puebla / Tlaxcala', 'Puebla')

#Tamaulipas
metro_by_region = metro_by_region.replace('Tamaulipas / Veracruz de Ignacio de la Llave', 'Tamaulipas')

# Assess the resulting number of Mexico States in dataset
# **Further Comments**: There are now the correct number of Mexican States in the metro_by_region dataset.
# Next, in order to smoothly merge *res_datatset* and the combined *metro_by_region*/*mexico_gdp* datastets,
# there must be at least one shared column name. Therefore, since *metro_by_region* and *res_dataset* share
# the same column purpose for State names, *res_dataset['Region']* will be renamed to *dates['State']* in order
# to match *metro_by_region*. Note: Additional modification to *res_dataset* will be applied as needed after the
# merging of *metro_by_region* and *mexico_gdp*.

# In res_dataset, rename 'Region' column to 'State' in order to match metro_by_region
res_dataset.columns = ['State',
                       'μ Age Maternal Mortality',
                       'Region (n)',
                       'μ Age Variance',
                       'Above(0) or Below(1) MEX μ',
                       'μ Region Education Level',
                       'μ Region Local Community Size',
                       'μ Presence(0)/Not(1) of Med Assist',
                       'μ Age Adolescent Maternal Death',
                       'Region Ado (n)']

# Part 2: Prepare Data for merging *metro_by_region['Name']* with *mexico_gdp['Metro Areas']*

# Assess Metropolitan Areas format in *mexico_gdp['Metro Areas']* dataset
metro_areas = mexico_gdp['Metro Areas'].sort_values()

# Assess Metropolitan Areas format in *metro_by_region['Name']* dataset
metro_areas_subdf = metro_by_region['Name'].sort_values()

# Observations:
# Some of the Metropolitan Area names in *metro_by_region['Name']* need to be cleaned to match
# *mexico_gdp['Metro Areas']* dataset. Namely:
# - 'Acapulco' = 'Acapulco de Juarez' 
# - 'Chilpancingo' = 'Chilpancingo de los Bravo'
# - 'Colima - Villa de Álvarez' = 'Colima'
# -  Culiacán = 'Culiacan'
# - 'Zacatecas - Guadalupe' = 'Guadalupe'
# - 'Juárez' = 'Juarez'
# - 'León' = 'Leon'
# - 'Mazatlán' = 'Mazatlan'
# - 'Valle de México\xa0[Greater Mexico City]' = 'Mexico City'
# - 'Minatitlán' = 'Minatitlan'
# - 'Monclova - Frontera' = 'Monclova'
# - 'Oaxaca' = 'Oaxaca de Juarez'
# - 'Pachuca' = 'Pachuca de Soto'
# - 'Poza Rica' = 'Poza Rica de Hidalgo'
# - 'Puebla - Tlaxcala' = 'Puebla'
# - 'Querétaro' = 'Queretaro' 
# - 'San Luis Potosí' = 'San Luis Potosi'
# - 'Tehuacán' = 'Tehuacan'
# - 'Tlaxcala - Apizaco' = 'Tlaxcala'
# - 'La Laguna\xa0(Comarca Lagunera, Torreón - Gómez Palacio)' = 'Torreon'
# - 'Tulancingo' = 'Tulancingo de Bravo'
# - 'Tuxtla Gutiérrez' = 'Tuxtla Gutierrez'

# #### Fix Metropolitan Names in metro_by_region['Name'] dataset
# - NOTE: The values not found in the mexico_gdp dataset will be skipped.

# Replace the metropolitan region names with the matching name in the GDP dataset

#Acapulco de Juarez
metro_by_region = metro_by_region.replace('Acapulco', 'Acapulco de Juarez')

#Chilpancingo de los Bravo
metro_by_region = metro_by_region.replace('Chilpancingo', 'Chilpancingo de los Bravo')

#Colima
metro_by_region = metro_by_region.replace('Colima - Villa de Álvarez', 'Colima')

#Culiacan
metro_by_region = metro_by_region.replace('Culiacán', 'Culiacan')

#Guadalupe
metro_by_region = metro_by_region.replace('Zacatecas - Guadalupe', 'Guadalupe')

#Juarez
metro_by_region = metro_by_region.replace('Juárez' , 'Juarez')

#Leon
metro_by_region = metro_by_region.replace('León' , 'Leon')

#Mazatlan
metro_by_region = metro_by_region.replace('Mazatlán' , 'Mazatlan')

#Mexico City
metro_by_region = metro_by_region.replace('Valle de México\xa0[Greater Mexico City]', 'Mexico City')

#Minatitlan
metro_by_region = metro_by_region.replace('Minatitlán' , 'Minatitlan')

#Monclova
metro_by_region = metro_by_region.replace('Monclova - Frontera', 'Monclova')

#Oaxaca de Juarez
metro_by_region = metro_by_region.replace('Oaxaca' , 'Oaxaca de Juarez')

#Pachuca de Soto
metro_by_region = metro_by_region.replace('Pachuca' , 'Pachuca de Soto')

#Poza Rica de Hidalgo
metro_by_region = metro_by_region.replace('Poza Rica' , 'Poza Rica de Hidalgo')

#Puebla
metro_by_region = metro_by_region.replace('Puebla - Tlaxcala', 'Puebla')

#Queretaro
metro_by_region = metro_by_region.replace('Querétaro' , 'Queretaro')

#San Luis Potosi
metro_by_region = metro_by_region.replace('San Luis Potosí' , 'San Luis Potosi')

#Tehuacan
metro_by_region = metro_by_region.replace('Tehuacán' , 'Tehuacan')

#Tlaxcala
metro_by_region = metro_by_region.replace('Tlaxcala - Apizaco', 'Tlaxcala')

#Torren
metro_by_region = metro_by_region.replace('La Laguna\xa0(Comarca Lagunera, Torreón - Gómez Palacio)', 'Torreon')
                                          
#Tulancingo de Bravo
metro_by_region = metro_by_region.replace('Tulancingo' , 'Tulancingo de Bravo')
                                          
#Tuxtla Gutierrez
metro_by_region = metro_by_region.replace('Tuxtla Gutiérrez' , 'Tuxtla Gutierrez')


# Verify results
metro_areas_subdf_2 = metro_by_region['Name'].sort_values()

# For Analysis of Population in average_age_maternal_mortality, make sub-population dataset

# Combine the Population values for each Mexican State
state_pop = metro_by_region.groupby(['State(s)']).sum()

# Reset index so States is a column
state_pop = state_pop.reset_index()

# Store state_pop dataset as a global variable that can be uploaded to other Jupyter Notebooks
# get_ipython().run_line_magic('store', 'state_pop')


# Drop all Metro Areas in metro_by_region not in mexico_gdp
condition = metro_by_region['Name'].isin(mexico_gdp['Metro Areas']) == True
metro_by_region['Drop if False'] = condition

# Remove all rows where metro_by_region['Drop if False'] == False
metro_by_region = metro_by_region[metro_by_region['Drop if False'] == True]

# Remove metro_by_region['Drop if False'] column
metro_by_region = metro_by_region.drop(columns=['Drop if False', 'Status'])
metro_areas_subdf = metro_by_region['Name'].sort_values()

# Sort metro_by_region Metro Areas in alphabetical order
metro_by_region = metro_by_region.sort_values(by='Name', ascending=True)

# Add a State Population columns to metro_by_region
# - Create variables for total population by State

# Create empty list variables for state population by year for metro_by_region

#2010
state_population_2010 = {}

#2015
state_population_2015 = {}

# Test code to create function
row = state_pop[state_pop['State(s)'] == 'Campeche']

# Add a total state population value for each metro area within a state
for state in state_pop['State(s)']:
    
    # Iterate over state_pop dataset
    for m_state in metro_by_region['State(s)']:
        if state == m_state:
            #store pop values of state_pop in metro_by_region
            row = metro_by_region[metro_by_region['State(s)'] == state]
            
            #2010
            p2010 = int(row['Population 2010'].sum())
            state_population_2010[state] = p2010
            
            #2015
            p2015 = int(row['Population 2015'].sum())
            state_population_2015[state] = p2015


metro_state_pop2010 = []
metro_state_pop2015 = []

for state in metro_by_region['State(s)']:
    for state10 in state_population_2010:
        if state == state10:
            
            #2010
            pop2010 = state_population_2010[state]
            metro_state_pop2010.append(pop2010)
            
            #2015
            pop2015 = state_population_2015[state]
            metro_state_pop2015.append(pop2015)

# Convert the state pop lists to a Series and add as new column to metro_by_region

#2010
metro_by_region['State Population 2010'] = metro_state_pop2010

#2015
metro_by_region['State Population 2015'] = metro_state_pop2015
m
# Remove metro_by_region['Population 2010', 'Population 2015'] columns as they are no longer needed
metro_by_region = metro_by_region.drop(columns=['Population 2010', 'Population 2015'])

# Rename 'Name' column to 'Metro Areas' in order to match mexico_gdp
metro_by_region.columns = ['Metro Areas', 'State', 'State Population 2010', 'State Population 2015']

# Remove Metro Areas in mexico_gdp that are not in metro_by_region
condition_2 = mexico_gdp['Metro Areas'].isin(metro_by_region['Metro Areas']) == True
mexico_gdp['Drop if False'] = condition_2

# Remove all rows where metro_by_region['Drop if False'] == False
mexico_gdp = mexico_gdp[mexico_gdp['Drop if False'] == True]

# Remove metro_by_region['Drop if False'] column
mexico_gdp = mexico_gdp.drop(columns=['Drop if False'])

# Rename 'Year_2010' and 'Year_2015' columns to 'GDP 2010' and 'GDP 2015' for clarity
mexico_gdp.columns = ['Metro GDP 2010', 'Metro GDP 2015', 'Metro Areas']
metro_areas_subdf = mexico_gdp['Metro Areas'].sort_values()

# Part 3: Merge *metro_by_region* with *mexico_gdp*
metro_gdp_merge = pd.merge(metro_by_region, mexico_gdp, on='Metro Areas')

# Part 4: Merge *metro_gdp_merge* with *res_dataset*
# - Since *res_dataset* and *metro_gdp_merge* have different lengths, the merging of the two datasets will most
# likely require an iteration over the rows of *res_dataset*

# Reindex dates by State
metro_gdp_merge = metro_gdp_merge.sort_values(by=['State'],ascending=True)
metro_gdp_merge = metro_gdp_merge.reset_index(drop=True)

# Remove States in *metro_gdp_merge* that are not in *res_dataset*
condition_3 = metro_gdp_merge['State'].isin(res_dataset['State']) == True
metro_gdp_merge['Drop if False'] = condition_3

# Remove all rows where metro_by_region['Drop if False'] == False
metro_gdp_merge = metro_gdp_merge[metro_gdp_merge['Drop if False'] == True]

# Remove metro_by_region['Drop if False'] column
metro_gdp_merge = metro_gdp_merge.drop(columns=['Drop if False'])

# Combine all Metro GDP 2010 and GDP 2015 data by State
vera = metro_gdp_merge[metro_gdp_merge['State'] == 'Veracruz de Ignacio de la Llave']

# *Preliminary Observation*: The values of Metro GDP appear to be strings. Therefore, the column values must
# be converted to integers prior to summing values.

# Convert GDP columns from strings to numbers

#2010
metro_gdp_merge['Metro GDP 2010'] = pd.to_numeric(metro_gdp_merge['Metro GDP 2010'], errors='coerce')

#2015
metro_gdp_merge['Metro GDP 2015'] = pd.to_numeric(metro_gdp_merge['Metro GDP 2015'], errors='coerce')


# Test code for the iteration 

#Aguascalientes
aqua = metro_gdp_merge[metro_gdp_merge['State'] == 'Aguascalientes']
#2010
aqua2010 = list(aqua['Metro GDP 2010'])
aqua2010 = sum(aqua['Metro GDP 2010'].astype('int64', errors='ignore'))
#2015
aqua2015 = list(aqua['Metro GDP 2015'])
aqua2015 = sum(aqua['Metro GDP 2015'].astype('int64', errors='ignore'))

#Veracruz de Ignacio de la Llave
vera = metro_gdp_merge[metro_gdp_merge['State'] == 'Veracruz de Ignacio de la Llave']
#2010
vera2010 = list(vera['Metro GDP 2010'])
vera2010 = sum(vera['Metro GDP 2010'].astype('int64', errors='ignore'))
#2015
vera2015 = list(vera['Metro GDP 2015'])
vera2015 = sum(vera['Metro GDP 2015'].astype('int64', errors='ignore'))

# Create empty dict variables for state gdp by year for metro_gdp_merge
#2010
state_gdp_2010 = {}
#2015
state_gdp_2015 = {}

# Add a total state gdp value for each metro area within a state
for state in metro_gdp_merge['State']:
    
    # select all rows for a given State
    subdf = metro_gdp_merge[metro_gdp_merge['State'] == state]
    
    # sum 2010 GDP
    gdp10 = list(subdf['Metro GDP 2010'])
    gdp10 = sum(subdf['Metro GDP 2010'].astype('int64', errors='ignore'))
    
    # append to state_population_2010 
    state_gdp_2010[state] = gdp10
    
    # sum 2015 GDP
    gdp15 = list(subdf['Metro GDP 2015'])
    gdp15 = sum(subdf['Metro GDP 2015'].astype('int64', errors='ignore'))
    
    # append to state_population_2015 
    state_gdp_2015[state] = gdp15

# Add State 2010 and 2015 GDP values to each row.

# Create empty list variables for state gdp by year for metro_gdp_merge
#2010
state_gdp2010 = []
#2015
state_gdp2015 = []

for state in metro_gdp_merge['State']:
    for state10 in state_gdp_2010:
        if state == state10:
            
            #2010
            gdp2010 = state_gdp_2010[state]
            state_gdp2010.append(gdp2010)
            
            #2015
            gdp2015 = state_gdp_2015[state]
            state_gdp2015.append(gdp2015)

# Convert the state gdp lists to a Series and add as new column to metro_by_region

#2010
metro_gdp_merge['State GDP 2010'] = state_gdp2010

#2015
metro_gdp_merge['State GDP 2015'] = state_gdp2015

# Remove metro_gdp_merge['Metro Areas', 'Metro GDP 2010', 'Metro GDP 2015'] columns as they are no longer needed
metro_gdp_merge = metro_gdp_merge.drop(columns=['Metro Areas', 'Metro GDP 2010', 'Metro GDP 2015'])

# Which Region GDP Increased between 2010 and 2015?
# **Create Binary Columns**:
# Region GDP 
# - 0 : Region GDP in 2015 *greater than or equal to* the Region's GDP in 2010
# - 1 : Region GDP in 2015  *less than* the Region's GDP in 2010

# Create a list item to hold comparison response
binary_gdp = []

# Create an iteration function to compare region mean to popupation mean
for state in metro_gdp_merge['State']:
    
    #select region row
    region = metro_gdp_merge[metro_gdp_merge['State'] == state] 
    
    gdp2010 = pd.to_numeric(np.nanmean(region['State GDP 2010']), errors='ignore')
    gdp2015 = pd.to_numeric(np.nanmean(region['State GDP 2015']), errors='ignore')
    
    if gdp2015 >= gdp2010:
        binary_gdp.append(0)
    else:
        binary_gdp.append(1)

# Convert the state gdp lists to a Series and add as new column 
metro_gdp_merge['Increase(0)/Not(1) GDP 2010-15'] = binary_gdp

# Part 4: Merge metro_gdp_merge with res_dataset
metro_gdp_mortality = pd.merge(metro_gdp_merge, res_dataset, on='State').drop_duplicates()

# Look over columns and column correlations: Are all columns in metro_gdp_mortality necessary?

# Create a correlation dataframe
feature_corr = metro_gdp_mortality.corr()

# Plot a correlation heatmap
sns.heatmap(feature_corr, square=True, cmap='RdYlGn')


# Further Data Cleaning:
# Since it appears that there is not correlation difference between columns with 2010 and 2015 values, so the 2010 values can be removed from the dataset:
#     - State Population
#     - State GDP 
# 
# Additionally, if 'Above(0)/Below(1) MEX μ' is the target variable, the following variables need to be removed because they correlate too much to the target variable and inevitably will give the Machine Learning Model the 'answer'
#     - μ Age Maternal Mortality
#     - Region (n)
#     - μ Age Variance
#     - μ Age Adolescent Maternal Death
#     - Region Ado (n)

#Remove columns as that are no longer needed
metro_gdp_mortality = metro_gdp_mortality.drop(columns=['State Population 2010', 
                                                        'State GDP 2010', 
                                                        'μ Age Maternal Mortality',
                                                        'Region (n)',
                                                        'μ Age Variance',
                                                        'μ Age Adolescent Maternal Death',
                                                        'Region Ado (n)'])

# Store merged metro_gdp_mortality dataset as a global variable
# get_ipython().run_line_magic('store', 'metro_gdp_mortality')
