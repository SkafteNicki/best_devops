Breast Cancer Classification

* This Python application demonstrates a simple machine learning workflow using the breast cancer dataset from the sklearn library. The application performs the following steps:

1.Load Dataset: The breast cancer dataset is loaded using the load_breast_cancer function from sklearn.datasets.

2.Split Dataset: The dataset is split into training and testing sets using an 80-20 split with train_test_split from sklearn.model_selection.

3.Standardize Features: The features are standardized using StandardScaler from sklearn.preprocessing to ensure that they have a mean of 0 and a standard deviation of 1.

4.Train Model: A Support Vector Machine (SVM) model with a linear kernel is trained using the standardized training set. The SVC class from sklearn.svm is used for this purpose.

5.Make Predictions: The trained SVM model is used to make predictions on the test set.

6.Evaluate Model: The performance of the model is evaluated using the accuracy_score and classification_report from sklearn.metrics. The accuracy score and a detailed classification report are printed, which include precision, recall, f1-score, and support for each class.

* How to Run
Ensure you have the necessary dependencies installed. You can install them using pip:

*pip install scikit-learn*

Run the script:

*python script_name.py*

Replace script_name.py with the actual name of your script file.

* TODO: improve the README.md file
* TODO: add `requirements.txt` file
* TODO: add `pyproject.toml` file
* TODO: add typing to the code
* TODO: add logging of results to files