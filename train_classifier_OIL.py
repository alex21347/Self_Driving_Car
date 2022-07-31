import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import pickle5 as pickle
from sklearn.linear_model import Ridge

X = np.genfromtxt('observations_data.csv', delimiter= ",")
y = np.genfromtxt('actions_data.csv', delimiter= ",")

print(X.shape)
print(y[0])

#accuracies_rf20 = []
#accuracies_rf200 = []
accuracies_rf500 = []

for i in tqdm(range(10)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i)

    #Random Forest20
    #rf20 = RandomForestRegressor(max_depth=20, random_state=i)
    #rf20.fit(X_train, y_train)

    #Random Forest200
    #rf200 = RandomForestRegressor(max_depth=20, n_estimators = 1000, random_state=i)
    #rf200.fit(X_train, y_train)   

    #Random Forest500
    rf500 = RandomForestRegressor(max_depth=20, n_estimators = 200,  random_state=i)
    rf500.fit(X_train, y_train)


    #y_pred_svm = svm.predict(X_test)
    #y_pred_rf20 = rf20.predict(X_test)
    #y_pred_rf200 = rf200.predict(X_test)
    y_pred_rf500 = rf500.predict(X_test)

    #accuracies_svm.append(accuracy_score(y_test, y_pred_svm))
    #accuracies_rf20.append(mean_squared_error(y_test, y_pred_rf20))
    #accuracies_rf200.append(mean_squared_error(y_test, y_pred_rf200))
    accuracies_rf500.append(mean_squared_error(y_test, y_pred_rf500))


filename = 'OIL_model_rf.sav'
pickle.dump(rf500, open(filename, 'wb'))
#loaded_model = pickle.load(open(filename, 'rb'))

#print(loaded_model.predict(X_test))

plt.figure()
#plt.hist(np.array(accuracies_rf20), density = True, alpha = 0.5, label = "Random Forest20")
#plt.hist(np.array(accuracies_rf200), density = True, alpha = 0.5, label = "Random Forest200")
plt.hist(np.array(accuracies_rf500), density = True, alpha = 0.5, label = "Random Forest500")
plt.xlabel("Average MSE")
plt.ylabel("density")
plt.legend()
plt.show()
#print("rf20 MSE : ", np.mean(np.array(accuracies_rf20)))
#print("rf200 MSE : ", np.mean(np.array(accuracies_rf200)))
print("rf500 MSE : ", np.mean(np.array(accuracies_rf500)))