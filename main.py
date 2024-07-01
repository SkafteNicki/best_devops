# Імпортуємо необхідні бібліотеки
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Додаємо нову функцію add з гілки new-branch-anyta
def add(a: int | float | np.ndarray, b: int | float | np.ndarray) -> int | float | np.ndarray:
    return a + b

def train_and_evaluate():
    # Завантажуємо набір даних
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Розділяємо набір даних на тренувальну і тестову вибірки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

    # Стандартизуємо ознаки
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Тренуємо модель SVM
    model = SVC(kernel='linear', random_state=8)
    model.fit(X_train, y_train)

    # Робимо передбачення на тестовій вибірці
    y_pred = model.predict(X_test)

    # Оцінюємо модель
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("I don`t know")
    print(f'Accuracy: {accuracy:.3f}')
    print('Classification Report:')
    print("Hello")
    print(report)
    
    return accuracy, report

if __name__ == "__main__":
    train_and_evaluate()

    print(add(1, 2))
    print(add(1.5, 2.2))
    print(add(np.array([1, 2]), np.array([3, 4])))
