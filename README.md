# deep-learning-challenge

# Model Report 
Explain the purpose of this analysis.
* This model is to show Alphabet Soup Nonprofit which applicants are going to be the best option for funding based on their chances of being a successful venture. This model will use deep learning and neural networks to analyze the features of the applicants. 

Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing
What variable(s) are the target(s) for your model? 
* Our target variable, y, is the IS_SUCCESSFUL column, that tells us whether or not the applicant is a successful choice. 
What variable(s) are the features for your model? 
* The feature set, X, is the dataset where we dropped/removed the IS_SUCCESSFUL column. 
What variable(s) should be removed from the input data because they are neither targets nor features? 
* I removed the EIS and NAME column. It was an identifier of the applicant, but not a target or a feature for the purpose of this model. These columns won't contribute to the machine learning. 

Compiling, Training, and Evaluating the Model
How many neurons, layers, and activation functions did you select for your neural network model, and why?
* I tried many different layers and functions, but was not able to find the perfect recipe to reach 75%. The best I was able to figure out was slightly over 73%. To achieve this, I had three hidden layers, with 8, 10, and 12 units, the first relu and the two others sigmoid activations. The outer layer was also sigmoid. I used 30 epochs when testing because I was trying so many combos, but used 50 to help try to increase my accuracy. 
Were you able to achieve the target model performance? 
* I was not able to achieve over 75% accuracy. I achieved slightly over 73%, which was still an improvement from the low 60s when I first started compiling the code. 
What steps did you take in your attempts to increase model performance?
    To improve my model's performance, I added additional layers and changed activation functions to increase the accuracy. I also adjusted the numebr of epochs and random states. 

Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
Another model that could also be used for this analysis would be random tree. The IS_SUCCESSFUL column can be treated as categorical data either 'yes' or 'no' if it will be successful, and this is compatible with the bindary formats of random tree. We could use a decision tree for this as well, since the success is either a yes or no category and so decision tree would fit, and perhaps random forest could work too using categorical data. I would first try decision tree to see if accuracy improves at all. 


# Instructions

# Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.
Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.
Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
What variable(s) are the target(s) for your model?
What variable(s) are the feature(s) for your model?
Drop the EIN and NAME columns.
Determine the number of unique values for each column.
For columns that have more than 10 unique values, determine the number of data points for each unique value.
Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
Use pd.get_dummies() to encode categorical variables.
Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

# Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.
Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
Create the first hidden layer and choose an appropriate activation function.
If necessary, add a second hidden layer with an appropriate activation function.
Create an output layer with an appropriate activation function.
Check the structure of the model.
Compile and train the model.
Create a callback that saves the model's weights every five epochs.
Evaluate the model using the test data to determine the loss and accuracy.
Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

# Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.
Use any or all of the following methods to optimize your model:
Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
Dropping more or fewer columns.
Creating more bins for rare occurrences in columns.
Increasing or decreasing the number of values for each bin.
Add more neurons to a hidden layer.
Add more hidden layers.
Use different activation functions for the hidden layers.
Add or reduce the number of epochs to the training regimen.
Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.
Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.
Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.
Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.
