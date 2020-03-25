import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    df = pd.read_csv('Data.csv')
    X = df.iloc[:, 0:5].values
    y = df.iloc[:, 5].values

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    file = open('model.pkl', 'wb')
    pickle.dump(clf, file)
    file.close()
    inputFeatures = [100, 1, 50, 1, 0]
    infProb = clf.predict_proba([inputFeatures])[0][1]
    print(infProb)