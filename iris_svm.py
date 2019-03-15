import pandas as pd
import numpy as np
data = pd.read_excel('irisdataset.xlsx')
x= data.iloc[:,0:4].values
y = data.iloc[:,4:5].values

from sklearn.preprocessing import LabelEncoder
y_en = LabelEncoder()
y = y_en.fit_transform(y)

import matplotlib.pyplot as plt
plt.scatter(x[:,1],x[:,2],c=y,s=50,cmap='autumn')
plt.show()

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.svm import SVC
obj_svc = SVC(kernel='linear',random_state=0)
obj_svc.fit(x_train,y_train)
y_pred = obj_svc.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)