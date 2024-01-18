Business Process Mining Project 
### by Ana Drmic and Leonardo Marazzi


The project involves embedding the ML framework for business process prediction and anomaly detection inside a notebook that can be used and re-used for both encoding the data and building a ML model.

The idea is to facilitate the building of different models, starting rom log encoding to model building, and then comparing them with each oher.
Our proposal is to have the notebook divided in to 4 parts:

1 Data Loading and Visualization

2 Encoding 

3 Model Building and Training 

4 Model Comparison

The model building has been implemented with an AutoML lybrary: h2o. This solution was chosen to build a viable framework to create models for business process mining and to find the best models without spending too much time on the parameter tuning.son.






The repository is divided into two main parts:

1) HMW: The notebooks for the homework. "hm_1" is the first homework with the small dataset, and "hm1_long" is the same homework with the larger dataset.

2) The main part of the project: The project is developed and commented on in the notebook "project," where we train the model for classification using aggregation encoding. In "project2" we demonstrate how the same pipeline works on a different dataset, and in "project3", we trained for the same problem as "project" but with index encoding.

In the Python file "function.py" you can find all the functions we developed, along with comments.
