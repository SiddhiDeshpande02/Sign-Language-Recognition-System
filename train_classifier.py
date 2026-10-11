import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

#data = np.asarray(data_dict['data'])
#labels = np.asarray(data_dict['labels'])

data = []
labels = []

for i in range(len(data_dict['data'])):
    sample = data_dict['data'][i]
    
    if len(sample) == 42:  # or 63 depending on your code
        data.append(sample)
        labels.append(data_dict['labels'][i])

data = np.asarray(data)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
