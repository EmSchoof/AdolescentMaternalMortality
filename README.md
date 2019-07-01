<h1>Springboard Capstone 1</h1>
<h1>Adolescent Maternal Mortality in Mexican States</h1>

<h3>Overview:</h3>
<p>In early 2019, I came across the the website <a href="https://dssg.uchicago.edu/2014/08/04/making-our-moms-proud-reducing-maternal-mortality-in-mexico/">Data Science for Good</a>, where a team of data scientists worked on a project known as 'Making our moms proud: Reducing Maternal Mortality in Mexico' that sought to assess why Mexico's Maternal Mortality Ratio (MMR, calculated by the WHO as # of deaths during pregnancy or within 42 days after birth , per 100,000 live birth) "has stagnated (over the past 10 years)despite additional efforts from the government to further bring it down."</p>
<p>As a woman in child-bearing years, it is both exhilerating and troubling to learn about the successes and failures in the advancements and understanding of women's health, especially when it comes to maternal mortality. My current residence in Southern California combined with my growing interest in becoming involved in Mexico-United States of America global relations, I decided to use the 'Making our moms proud' dataset as a learning tool for my first capstone project for Springboard's Data Science Career Track.</p>   
<p>Conclusion: 
Creating a Logistic Regression Model off of the scaled data produced a far more accurate predictive model than the unscaled data. Therefore, based off of the scaled dataset, the machine learning model created was accurately able to predict if a Region in Mexico will have a mean age of maternal mortality that is above or below the country's mean age, based off of the Region's GDP, recent changes in GDP, population size, mean educational level of maternal women, and the mean average of presence of medical assistance for maternal women. 

As stated earlier, the primary purpose of this study was to assess some of the economic factors the relate to higher levels of adolescent maternal mortality (complications for young women and girls below the age of 20) in Mexico. Worldwide, over 13 million adolescent girls give birth every year, and complications from those pregnancies and childbirth are a leading cause of death for those young mothers. It should be noted that simply detecting that a Region's mean age of maternal mortality falls below the country's norm does not solve this issue. Rather, having a mean age that is below the norm can be used as an indicator that that region may have higher instances of adolescent maternal mortality and where more aid is needed, especially with regions with lower averages of availability to healthcare.

Additional machine learning models that incorporate the level sex education, average distance from the nearest hospital, and number of child-bride instances within each Region of Mexico can help provide additional, more detailed information on the likelihood of adolescent maternal mortality. Assessing these factors can provide correlation data to potential needed resources (such as increased access to healthcare) and socioeconomic factors (such as child brides) that provide measurable factors to quantify a further reduction the rate of young mother mortality.</p>

<p>The scripts provided here calculate the risk probability of an adolescent maternal mortality by State in Mexico based on some of the top features contributing to maternal mortality. While this is a multi-dimensional issue, for the sake of this study, the following 8 factors were used to predict the likelihood of adolescent maternal mortality by region: region population, region GDP, local poverty level, level of education, and access to medical assistance in order to help direct government funds to areas where it would be most beneficial. </p>
<h3>Use</h3>
<ul>
    <li><a href="https://github.com/EmSchoof/Capstone-Project-1/blob/master/translation_english.txt">translation_english.txt</a> translates Spanish columns and data information into its English counterpart.</li>  
     <li><a href="https://github.com/EmSchoof/Capstone-Project-1/blob/master/maternal_mortality.ipynb">maternal_mortality.ipynb</a> cleans the source data and assesses mean maternal mortality age and mean adolescent maternal mortality age by State in Mexico.</li>  
     <li><a href="https://github.com/EmSchoof/Capstone-Project-1/blob/master/merging_dataframes.ipynb">merging_dataframes.ipynb</a> merges averaged information on State instances of maternal mortality with State enconomic factors (GDP and Population Size).</li>  
     <li><a href="https://github.com/EmSchoof/Capstone-Project-1/blob/master/machine_learning.ipynb">machine_learning.ipynb</a> produces the linear regression model assessing likelihood of adolescent maternal mortality within a State of Mexico.</li>  
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
