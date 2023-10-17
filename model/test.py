import pickle
path='random_forest_model.pkl'
f=open(path, 'rb')
data=pickle.load(f)
print(data)
