import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import pickle5 as pickle

#import xgboost as xgb

X_big = np.genfromtxt('data/bigcones.txt', delimiter= ", ")
X_small = np.genfromtxt('data/smallcones.txt', delimiter= ", ")
X_big = np.concatenate((X_big, np.ones((len(X_big[:,0]), 1))), axis = 1)
X_small = np.concatenate((X_small, np.zeros((len(X_small[:,0]), 1))), axis = 1)


print(X_big[0,:])
print(X_small[0,:])

#plt.figure()

#plt.hist(X_big[:,0], alpha = 0.5)
#plt.hist(X_small[:,0], alpha = 0.5)

#plt.show()

#plt.figure()

#plt.hist(X_big[:,1], alpha = 0.5)
#plt.hist(X_small[:,1], alpha = 0.5)

#plt.show()

#plt.figure()

#plt.hist(X_big[:,2], alpha = 0.5)
#plt.hist(X_small[:,2], alpha = 0.5)

#plt.show()

X = np.concatenate((X_small, X_big), axis = 0)
y = X[:,3]
X = X[:,:3]
#accuracies_svm = []
accuracies_rf3 = []
accuracies_rf4 = []
for i in tqdm(range(1)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=i)

    #SVM
    #svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    #svm.fit(X_train, y_train)

    #Random Forest
    rf4 = RandomForestClassifier(max_depth=4, random_state=i)
    rf4.fit(X_train, y_train)

    #Random Forest
    rf3 = RandomForestClassifier(max_depth=3, random_state=i)
    rf3.fit(X_train, y_train)

    #y_pred_svm = svm.predict(X_test)
    y_pred_rf4 = rf4.predict(X_test)
    y_pred_rf3 = rf3.predict(X_test)

    #accuracies_svm.append(accuracy_score(y_test, y_pred_svm))
    accuracies_rf4.append(accuracy_score(y_test, y_pred_rf4))
    accuracies_rf3.append(accuracy_score(y_test, y_pred_rf3))
    print(accuracy_score(y_test, y_pred_rf4))


filename = 'cone_classifier_rf.sav'
pickle.dump(rf4, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))

#print(loaded_model.predict(X_test))

#plt.figure()
##plt.hist(np.array(accuracies_svm), density = True, alpha = 0.5, label = "SVM")
#plt.hist(np.array(accuracies_rf4), density = True, alpha = 0.5, label = "Random Forest")
#plt.hist(np.array(accuracies_rf3), density = True, alpha = 0.5, label = "Random Forest")
#plt.xlabel("Accuracy")
#plt.ylabel("density")
#plt.legend()
#plt.show()

##print("svm mean accuracy : ", np.mean(np.array(accuracies_svm)))
#print("rf4 mean accuracy : ", np.mean(np.array(accuracies_rf4)))
#print("rf3 mean accuracy : ", np.mean(np.array(accuracies_rf3)))