
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate():
    # TODO: add arguments and argument parsing for high-level configuration

    # Load the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets
    # TODO: consider using cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Support Vector Machine (SVM) model
    # TODO: consider using a different model
    # TODO: consider hyperparameter tuning
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("HELLO SATANAS")

    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)
    return accuracy, report

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate an SVM on the breast cancer dataset.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--kernel', type=str, default='linear', help='Kernel type to be used in the SVM.')
    parser.add_argument('--log_file', type=str, default='results.log', help='File to log results.')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(filename=args.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    train_and_evaluate(test_size=args.test_size, random_state=args.random_state, kernel=args.kernel)

if __name__ == "__main__":
    train_and_evaluate()