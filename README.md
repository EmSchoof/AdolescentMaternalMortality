<h1>Springboard Capstone 1</h1>
<h1>Adolescent Maternal Mortality in Mexican States</h1>

<h3>Overview:</h3>
<p>In early 2019, I came across the the website <a href="https://dssg.uchicago.edu/2014/08/04/making-our-moms-proud-reducing-maternal-mortality-in-mexico/">Data Science for Good</a>, where a team of data scientists worked on a project known as 'Making our moms proud: Reducing Maternal Mortality in Mexico' that sought to assess why Mexico's Maternal Mortality Ratio (MMR, calculated by the WHO as # of deaths during pregnancy or within 42 days after birth , per 100,000 live birth) "has stagnated (over the past 10 years)despite additional efforts from the government to further bring it down."</p>
<p>As a woman in child-bearing years, it is both exhilerating and troubling to learn about the successes and failures in the advancements and understanding of women's health, especially when it comes to maternal mortality. My current residence in Southern California combined with my growing interest in becoming involved in Mexico-United States of America global relations, I decided to use the 'Making our moms proud' dataset as a learning tool for my first capstone project for Springboard's Data Science Career Track.</p>   
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
