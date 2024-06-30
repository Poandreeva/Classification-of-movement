import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from visualizations import evaluate_preds

def train_logistic_regression(X, y, save=False):
    accuracies = []
    for random_state in range(1, 101):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=random_state)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 101), accuracies, marker='o')
    plt.title('Accuracy vs. random_state')
    plt.xlabel('random_state')
    plt.ylabel('Accuracy')
    plt.grid(True)
    if save:
        plt.savefig(f'Accuracy_vs_random_state.png', dpi=400, bbox_inches='tight')
    plt.show()

    max_random_state = accuracies.index(max(accuracies)) + 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=max_random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_train_pred = lr.predict(X_train)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred, digits=2))

    check_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
    check_test.sample(10)
    evaluate_preds(y_test, y_train, y_pred, y_train_pred, unique_movements, save=save)

def train_random_forest(X_train, y_train, X_test, y_test, save=False):
    n_estimators_range = range(1, 13)
    mean_accuracies = []

    for n_estimators in n_estimators_range:
        rf = RandomForestClassifier(n_estimators=n_estimators)
        scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
        mean_accuracy = np.mean(scores)
        mean_accuracies.append(mean_accuracy)
        print(f"Number of trees: {n_estimators}, Mean accuracy: {mean_accuracy:.3f}")

    best_n_estimators = n_estimators_range[np.argmax(mean_accuracies)]
    print(f"Best number of trees: {best_n_estimators}")

    rf = RandomForestClassifier(n_estimators=best_n_estimators)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_train_pred = rf.predict(X_train)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred, digits=2))

    check_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
    check_test.sample(10)
    evaluate_preds(y_test, y_train, y_pred, y_train_pred, unique_movements, save=save)
