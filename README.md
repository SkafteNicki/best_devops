# SVM Classifier for Breast Cancer Dataset

This repository contains a simple Python script for training and evaluating a Support Vector Machine (SVM) classifier on the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn.

## Features

- Cross-validation: Uses 5-fold cross-validation for robust model evaluation.
- Hyperparameter tuning: Employs GridSearchCV to find optimal hyperparameters.
- Argument parsing:  Allows for flexible configuration through command-line arguments.
- Detailed output:  Prints cross-validation scores, accuracy, and classification report.

## Usage

1. Install Dependencies:
   pip install -r requirements.txt
2. Run the script:
   
   python main.py --test_size 0.3 --kernel rbf --random_state 10
   

   - --test_size: Size of the test set (default: 0.2).
   - --kernel: Kernel type for SVM (default: 'linear').
   - --random_state: Random seed for splitting (default: 42).

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.