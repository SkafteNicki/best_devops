# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate(test_size=0.2, use_cross_validation=False, model_type='svm', kernel='linear', use_poly_features=False, degree=2, tune_hyperparameters=False):
    # TODO: add arguments and argument parsing for high-level configuration

    # Load the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets
    # TODO: consider using cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    # TODO: consider better feature engineering
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_type == 'svm':
        model = SVC(kernel=kernel, random_state=42)
        if tune_hyperparameters:
            from sklearn.model_selection import GridSearchCV
            param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
            model = GridSearchCV(SVC(), param_grid, refit=True, verbose=0)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        if tune_hyperparameters:
            from sklearn.model_selection import GridSearchCV
            param_grid = {'n_estimators': [10, 50, 100], 'max_features': ['auto', 'sqrt', 'log2']}
            model = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=0)
    else:
        raise ValueError("Unsupported model type")

    # Train a Support Vector Machine (SVM) model
    # TODO: consider using a different model
    # TODO: consider hyperparameter tuning
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    # TODO: consider using more evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)
    return accuracy, report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a machine learning model.")
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--use_cross_validation', action='store_true', help='Whether to use cross-validation.')
    parser.add_argument('--model_type', type=str, default='svm', help='Type of model to use (e.g., "svm" or "random_forest").')
    parser.add_argument('--kernel', type=str, default='linear', help='Kernel type to be used in the SVM model.')
    parser.add_argument('--use_poly_features', action='store_true', help='Whether to use polynomial features.')
    parser.add_argument('--degree', type=int, default=2, help='Degree of polynomial features.')
    parser.add_argument('--tune_hyperparameters', action='store_true', help='Whether to perform hyperparameter tuning.')

args = parser.parse_args()
train_and_evaluate(test_size=args.test_size, use_cross_validation=args.use_cross_validation, model_type=args.model_type, kernel=args.kernel, use_poly_features=args.use_poly_features, degree=args.degree, tune_hyperparameters=args.tune_hyperparameters)