# Artificial Neural Network
This project implements a multilayer artificial neural network fromscratch in Python. 
The framework is used for binary classification on two datasets:
* [Glassdoor Data](http://bit.ly/GlassdoorApply):

  Imagine a user visiting Glassdoor, and performing a job search. From the set of displayed results, user clicks on certain ones that she is interested in, and after checking job descriptions, she further clicks on apply button therein to land in to an application page. The apply rate is defined as the fraction of applies (after visiting job description pages), and the goal is to predict this metric using the above dataset (provided by a lead data scientist @ Glassdoor).
  
  Each row in the dataset corresponds to a user’s view of a job listing. It has 11 columns as described below:                     
      
      1. title_proximity_tfidf: Measures the closeness of query and job title.  
      2. description_proximity_tfidf: Measures the closeness of query and job description.                                                           
      3. main_query_tfidf: A score related to user query closeness to job title and job description.                                                     
      4. query_jl_score: Measures the popularity of query and job listing pair. 
      5. query_title_score: Measures the popularity of query and job title pair.
      6. city_match: Indicates if the job listing matches to user/user-specified location.                                                            
      7. job_age_days: Indicates the age of job listing posted.                 
      8. apply: Indicates if the user has applied for this job listing.         
      9. search_date_pacific: Date of the activity.                             
      10. u_id: ID of user (for privacy reasons ID is anonymized).              
      11. mgoc_id: Class ID of the job title clicked.                           
                                                                              
    Training set: The examples with the “search date pacific” column (9-th column)between 01/21/2018-01/26/2018.                               
    
    Test set: The examples with the “search date pacific” column (9-th column) on 01/27/2018.                                                
    Inputs to the input layer::Features: First 7 columns.
    Label {0,1}: 8th column, i.e., apply.

* [Mammography Data](http://bit.ly/MammoData):
  
  Goal: Predict the severity (benign or malignant) of a mammographic mass lesion from BI-RADS attributes and the patient's age.
    Number of Features: 5 (1 non-predictive, 4 predictive)                 
    Attribute Information:                                                 
        
        1. BI-RADS assessment: 1 to 5 (ordinal)                            
        2. Age: patient's age in years (integer)                                  
        3. Mass shape: ound=1 oval=2 lobular=3 irregular=4 (nominal)              
        4. Mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)                                    
        5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)  
        6. Severity: benign=0 or malignant=1 (binominal)                          
    
    Obtained from <https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass>
## Network Architecture
We implement a multi-layer neural network with a single hidden layer with full feed-forward interconnection (units in one layer are connected to every unit in the next) in order to learn nonlinear decision boundaries. There are ![Equation](https://latex.codecogs.com/gif.latex?h) neurons in the hidden layer. Note that given an adequate number of hidden units ![Equation](https://latex.codecogs.com/gif.latex?h), networks with a single hidden layer can implement arbitrary decision boundaries and any higher number of layers makes training times become very long. The activation function in both hidden and output layers is the Sigmoid activation (it's possible to apply ReLU activation function to the nodes in the hidden layer).

Note that in both datasets, the _numerical_ features are standardized to have zero mean and unit standard deviation and examples (rows) with missing values are removed.

### Analysis
In evaluating the performance of the classifier, care must be taken in terms of selecting the appropriate metric.   
* Glassdoor Dataset
   
  In this dataset, the + class seems to be a lot less probable than the − class. In this case, a classifier can achieve good accuracy by always predicting the − class, obviously indicating a non-ideal classifier, i.e., accuracy ignores the probability estimations of classification in favor of class labels. Receiver operating characteristics (ROC) curve, on the other hand, shows the trade-off between false positive and true positive rates, making the area below the ROC curve (AUC) a better measure than accuracy for our scenario. In fact, ROC describes the discriminative power of a classifier independent of class distribution and unequal prediction error costs.

  Ideally, a classifier has a true positive rate of 1 and a false positive rate of 0 (AUC= 1), and hence a classifier is better the closer it gets to the upper-left corner. On the ![Equation](https://latex.codecogs.com/gif.latex?45%5E%5Cdegree) baseline, we have the worst performance as we make as many true decisions as false ones (any classifier that is below the diagonal can be improved by flipping its decision).
        
  We illustrate the advantage of the AUC over accuracy by plotting the ROC curve as well as accuracy versus ![Equation](https://latex.codecogs.com/gif.latex?%5Ctheta). 
  
  As we can see, the accuracy metric is excellent for a large enough value of ![Equation](https://latex.codecogs.com/gif.latex?%5Ctheta), i.e., when predicting every instance as −.
  
  Since the classes are very imbalanced, we also plot the Precision-Recall curve; precision is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned.
* Mammography Dataset

  This dataset is more or less balanced as shown by the baseline of the Precision-Recall curve, i.e., ![Equation](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cfrac%7BP%7D%7BP&plus;N%7D), where P is the number of positive labels in the traning set and N is the number of negative labels.
