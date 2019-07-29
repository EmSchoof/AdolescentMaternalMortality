<h1>Springboard Capstone 1</h1>
<h1>Likelihood of Adolescent Maternal Mortality in Mexican States</h1>

<h3>Background:</h3>
<p>In early 2019, I was traveling through various parts of Latin America around the same time that my Springboard course began. When my Mentor ended up being from Mexico, I decided that I wanted to work on a project that would help give back to my home country's Latin neighbor, and to visit the country during my studies. I came across the the website <a href="https://dssg.uchicago.edu/2014/08/04/making-our-moms-proud-reducing-maternal-mortality-in-mexico/">Data Science for Good</a>, where a team of data scientists worked on a project that sought to assess why Mexico's Maternal Mortality Ratio (MMR, calculated by the WHO as # of deaths during pregnancy or within 42 days after birth , per 100,000 live birth) "has stagnated (over the past 10 years)despite additional efforts from the government to further bring it down." As an American woman in child-bearing years traveling in Latin America, focusing on the maternal mortality factors present in Mexico seemed like a fitting topic to analyze and design my first capstone project. This dataset was used as the primary source of features for the study.</p>

<p>It is both exhilerating and troubling to learn about the successes and failures in the advancements in women's health worldwide, especially when it comes to maternal mortality. Based off of the World Health Organization's website, maternal mortality is usually the result of preventable complications during pregnancy and the act of childbirth. It would make sense, then, that maternal mortality is more common in rural and poor communities lacking access to care and resources to help pregnant women. Even more alarming, however, is the high prevalence of adolescent maternal mortality (ages less than 20), who face a higher risk of complications and mortality as a result of pregnancy than other women. Worldwide, over 13 million adolescent girls give birth every year, and complications from those pregnancies and childbirth are a leading cause of death for those young mothers.</p>  

<h3>Overview:</h3>
<p>The purpose of this particular study is to assess the factors that impact the likelihood of adolescent maternal mortality within each state of Mexico. It should be noted that simply detecting that a Region's mean age of maternal mortality falls below the country's norm does not solve this issue. Rather, having a mean age that is below the norm can be used as an indicator that the region may have higher instances of adolescent maternal mortality and where more aid is needed, especially with regions with lower averages of availability to healthcare. Within the Data Science for Good dataset, I chose to mimic the factors listed on the WHO's webpage explaining maternal mortality and selected rows for each woman's age at mortality, local community size, education level reached, and presence of medical assistance. Since these were mostly numerical values, that made it possible to averaged these values to represent the 'average maternal mortality by state'. Additional information about the state's overall GDP and population size was then merged with the averaged features within the Data Science for Good datatset.</p>

<p>Important Factors of Interest:</p>
<ul>
  <li>Mean Age of Maternal Mortality by State in Mexico</li>
  <li>GDP by State in Mexico</li>
  <li>Population by State in Mexico</li>
  <li>Length of Women Education</li>
  <li>Presence of Medical Assistance</li>
  <li>Region Local Community Size</li>
</ul>
    
<h3>Model Construction:</h3>
<p><b>Dependent (Target) Variable:</b> A binary variable indicating if the state's mean age of maternal mortality was Above(0) or Below(1) Mexico's overall mean age of maternal mortality.</p>
<p><b>Independent (Feature) Variables:</b></p>
<ul>
  <li>State<li>
  <li>State GDP in 2015</li>
  <li>State Population Size in 2015</li>
  <li>If State GDP Increase(0)/Not(1) from 2010 to 2015</li>
  <li>State Average Length of Education in Deceased Maternal Women</li>
  <li>State Average Presence of Medical Assistance Received by Deceased Maternal Women</li>
  <li>State Average Local Community Size of Deceased Maternal Women</li>
</ul>

<h3>Conclusion:</h3>
<p>Creating a Logistic Regression Model off of the scaled data produced a far more accurate predictive model than the unscaled data. Therefore, based off of the scaled dataset, the machine learning model created was accurately able to predict if a Region in Mexico will have a mean age of maternal mortality that is above or below the country's mean age, based off of the Region's GDP, recent changes in GDP, population size, mean educational level of maternal women, mean local community sizes of maternal women within each region, and the mean average of presence of medical assistance for maternal women.</p>

<p>Additional machine learning models that incorporate the level sex education, average distance from the nearest hospital, and number of child-bride instances within each Region of Mexico can help provide additional, more detailed information on the likelihood of adolescent maternal mortality. Assessing these factors can provide correlation data to potential needed resources (such as increased access to healthcare) and socioeconomic factors (such as child brides) that provide measurable factors to quantify a further reduction the rate of young mother mortality.</p>

<p>The scripts provided here calculate the risk probability of an adolescent maternal mortality by State in Mexico based on some of the top features contributing to maternal mortality. While this is a multi-dimensional issue, for the sake of this study, the following 8 factors were used to predict the likelihood of adolescent maternal mortality by region: region population, region GDP, local poverty level, level of education, and access to medical assistance in order to help direct government funds to areas where it would be most beneficial. </p>

<h3>Use</h3>
<ul>
    <li><a href="https://github.com/EmSchoof/AdolescentMaternalMortality/blob/master/data/translation_english.txt">translation_english.txt</a> translates Spanish columns and data information into its English counterpart.</li>  
     <li><a href="https://github.com/EmSchoof/AdolescentMaternalMortality/blob/master/python_files/1_data_wrangling.ipynb">1_data_wrangling.ipynb</a> cleans the source data and assesses mean maternal mortality age and mean adolescent maternal mortality age by State in Mexico.</li>  
     <li><a href="https://github.com/EmSchoof/AdolescentMaternalMortality/blob/master/python_files/2_explanatory_data_analysis.ipynb"> 2_explanatory_data_analysis.ipynb</a>Assesses distribution of general target variable.</li>  
     <li><a href="https://github.com/EmSchoof/AdolescentMaternalMortality/blob/master/python_files/3_adomaternal_mortality.ipynb"> 3_adomaternal_mortality.ipynb</a>Assesses distribution of just adolescent maternal mortality.</li>  
   <li><a href="https://github.com/EmSchoof/AdolescentMaternalMortality/blob/master/python_files/4_inferential_statistics.ipynb">4_inferential_statistics.ipynb</a>Statistically proves mean age of maternal mortality in the dataset is comparable to the actual mean age of maternal mortality in Mexico. Also proves via ANOVA that at least one state in Mexico has a mean age of maternal mortality statistically different from the others.</li>  
</ul>
   <li><a href="https://github.com/EmSchoof/AdolescentMaternalMortality/blob/master/python_files/5_merging_dataframes.ipynb">5_merging_dataframes.ipynb</a> Merges averaged information on State instances of maternal mortality with State enconomic factors (GDP and Population Size).</li>  
</ul>
   <li><a href="https://github.com/EmSchoof/AdolescentMaternalMortality/blob/master/python_files/6_machine_learning.ipynb">6_machine_learning.ipynb</a>Production of the pipeline machine learning model using logistic regression and standard scaling to assess the likelihood of adolescent maternal mortality within a State of Mexico.</li>  
</ul>

<h3>Dependencies</h3>
The specific Python files written by this code assume you have the following tools added to your module directories:
<br>
<ul style="list-style-type:circle;">
    <li>General</li>
        <ul>
            <li>pandas</li>
            <li>numpy</li>
            <li>seaborn</li>
        </ul>

<li>Data Visualization</li>
        <ul>
            <li>pyplot</li>
            <li>pylab</li>
            <li>empirical_distribution</li>
        </ul>

 <li>Statistics</li>
        <ul>
            <li>statistics</li>
            <li>scipy</li>
        </ul>

<li>Machine Learning modules</li>
        <ul>
            <li>train_test_split</li>
            <li>DecisionTreeClassifier</li>
            <li>RandomForestClassifier</li>
            <li>LogisticRegression</li>
        </ul>

 <li>Hyperparameters</li>
        <ul>
            <li>GridSearchCV</li>
            <li>roc_auc_score</li>
            <li>cross_val_score</li>
        </ul>

<li>ROC Curve</li>  

<li>Scale Data</li>
        <ul>
            <li>StandardScaler</li>
        </ul>
</ul>
