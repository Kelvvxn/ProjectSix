
## Name

Kelvin Peterson

## Files and Uses:

* GlassIdentifier.ipynb: This notebook analyzes glass identification using chemical composition data from the UCI Machine Learning Repository. It includes data preprocessing, feature analysis, and machine learning (Decision Tree) to classify different types of glass based on their chemical properties.  A GUI component is included for interactive data exploration, searching, and prediction.
* README.txt/mb: This file explains the purpose, functionality, and usage of the project.

## Dependencies:

Before installing and running the program, ensure you have a compatible operating system like Windows, macOS, or Linux. You also need Python Version 3+.  Also make sure all the files are in the same directory. The user will have to download the following libraries to run the code:

* numpy
* pandas
* scipy
* matplotlib
* scikit-learn

## Version History:

* 4/27-28: Found the UCI 
* 4/29: Wrote the code to do the data quality check, "clean" the data, and visualize it.
* 5/1: Wrote the code to do the descision tree.
* 5/2-5/3: Worked on a wrote the code that does the GUI. At first I solidified the ability to be able to have the gui filter by a specific classification or just show all. Then after that I worked on the _Search Class Data_ part of the gui that allows the user to search a number that is apart of the chemical composition of a certain type of glass and it will appear, or they can type a number and a certain type of glass will appear. I then tried lastly to see if I could make code that would predict a glass type based on what the user would input for but it ended up not working and as of 5/3 the code to predict glass does not work.

## Notes About Output:

There is a lot that occurs in the output that should be explained. When the code is run in something like an ide, a lot is produced and this part will be to explain it.

The first part is the Data Quality Check:
=== Data Quality Check ===
Data Shape: (214, 10)

Data Types:
RI               float64
Na               float64
Mg               float64
Al               float64
Si               float64
K                float64
Ca               float64
Ba               float64
Fe               float64
Type_of_glass      int64
dtype: object

Missing Values:
RI               0
Na               0
Mg               0
Al               0
Si               0
K                0
Ca               0
Ba               0
Fe               0
Type_of_glass    0
dtype: int64

Duplicate Rows: 1

This basically first tells you the size of the data with the rows being the amount of glass samples and the columns being attributes of the class.
The data types tells you what type of data type is stored and this case it is either a float or an int.
The missing values tell you if any of the chemical compostion of code in any of the rows are missing and values and the duplicate rows tells you if any 
of the data is the same. In this case it doesn't mean anything and is possible the similar glass was sampled.

---------------------------------------------------------------------------------------------------------------------------------------------------------------

The next part is the summary of the statistics.

Summary Statistics:
               RI          Na          Mg          Al          Si           K  \
count  214.000000  214.000000  214.000000  214.000000  214.000000  214.000000   
mean     1.518365   13.407850    2.684533    1.444907   72.650935    0.497056   
std      0.003037    0.816604    1.442408    0.499270    0.774546    0.652192   
min      1.511150   10.730000    0.000000    0.290000   69.810000    0.000000   
25%      1.516522   12.907500    2.115000    1.190000   72.280000    0.122500   
50%      1.517680   13.300000    3.480000    1.360000   72.790000    0.555000   
75%      1.519157   13.825000    3.600000    1.630000   73.087500    0.610000   
max      1.533930   17.380000    4.490000    3.500000   75.410000    6.210000   

               Ca          Ba          Fe  Type_of_glass  
count  214.000000  214.000000  214.000000     214.000000  
mean     8.956963    0.175047    0.057009       2.780374  
std      1.423153    0.497219    0.097439       2.103739  
min      5.430000    0.000000    0.000000       1.000000  
25%      8.240000    0.000000    0.000000       1.000000  
50%      8.600000    0.000000    0.000000       2.000000  
75%      9.172500    0.000000    0.100000       3.000000  
max     16.190000    3.150000    0.510000       7.000000

Class Distribution:
Type_of_glass
2    76
1    70
7    29
3    17
5    13
6     9
Name: count, dtype: int64

=== Data Cleaning ===
Outliers in RI: 17
Outliers in Na: 7
Outliers in Mg: 0
Outliers in Al: 18
Outliers in Si: 12
Outliers in K: 7
Outliers in Ca: 26
Outliers in Ba: 38
Outliers in Fe: 12

Summary Statistics is where you get a statistical overview of your numerical data.
For each numerical column (RI, Na, Mg, Al, Si, K, Ca, Ba, Fe), it gives you:
count: How many values are in the column (214 for all, which matches your total rows).
mean: The average value of that column.
std: The standard deviation (how spread out the values are).
min: The smallest value in the column.
25%: The 25th percentile (the value below which 25% of the data falls).
50%: The median (the middle value).
75%: The 75th percentile (the value below which 75% of the data falls).
max: The largest value in the column.
Example:
For the 'RI' column: The average refractive index is 1.518, the values range from 1.511 to 1.533, etc.

Class Distribution shows how many samples you have for each type of glass.
For example:
76 samples are of glass type 2.
70 samples are of glass type 1.
Only 9 samples are of glass type 6.
This helps you see if your data is "balanced" (roughly equal numbers of each type) or "imbalanced" (some types are much more common).

Data Cleaning reports the number of outliers found in each chemical property column.
Outliers are data points that lie significantly far from other data points. The code uses the Interquartile Range (IQR) method to identify them.
For example, it states "Outliers in RI: 17", meaning 17 data points were considered outliers in the 'RI' (Refractive Index) column.
In this case the code does not remove these outliers. This is a crucial decision in this context because outliers in chemical composition could be important indicators of specific glass types

---------------------------------------------------------------------------------------------------------------------------------------------------------------

Decision Tree Model Evaluation:

Accuracy: 0.9846153846153847

Classification Report:
              precision    recall  f1-score   support

           1       1.00      1.00      1.00        19
           2       1.00      1.00      1.00        23
           3       1.00      1.00      1.00         4
           5       1.00      0.83      0.91         6
           6       0.75      1.00      0.86         3
           7       1.00      1.00      1.00        10

    accuracy                           0.98        65
   macro avg       0.96      0.97      0.96        65
weighted avg       0.99      0.98      0.99        65


Cross-validation scores: [1.         1.         1.         1.         0.97619048]
Mean CV Score: 0.9952380952380953

Lastly is the Decision Tree Model Evaluation. This is an example output from when I ran the code:

The first part is the accuracy of the Decision Tree model on the test set. It means the model correctly predicted the glass type for approximately 98.5% of the samples in the test set.

Classification Report provides a more detailed breakdown of the model's performance for each glass type.

Key metrics:
precision: For each glass type, what proportion of the samples predicted as that type actually belong to that type? High precision means the model is good at avoiding false positives.
recall: For each glass type, what proportion of the samples that actually belong to that type were correctly predicted? High recall means the model is good at avoiding false negatives.
f1-score: The harmonic mean of precision and recall, balancing both metrics. It's a good single metric to consider when you want a balance between precision and recall.
support: The number of samples in the test set for each glass type.
Example Interpretation:
For glass type '1', precision, recall, and f1-score are all 1.00. This means the model perfectly classified all samples of type 1.
For glass type '5', precision is 1.00, but recall is 0.83. This indicates that while the model was correct every time it predicted a sample as type 5, it missed 17% of the actual type 5 samples.
The 'accuracy', 'macro avg', and 'weighted avg' lines provide overall summaries of these metrics. 'Macro avg' is the simple average across all classes, while 'weighted avg' weights the metric by the number of samples in each class.

Cross-validation scores: [1. 1. 1. 1. 0.97619048]

Cross-validation is a technique to assess how well a model generalizes to unseen data. The data is split into multiple "folds," and the model is trained and evaluated multiple times, each time using a different fold as the test set.   
These are the accuracy scores from each of the 5 folds in the cross-validation.
Mean CV Score: 0.9952380952380953

This is the average of the cross-validation scores. It gives a more robust estimate of the model's performance than a single train-test split accuracy.

At the end it also print a decision tree graph to show how the model comes to the conclusion for what it selects.

