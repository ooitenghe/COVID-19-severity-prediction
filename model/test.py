import pickle
path='random_forest.pkl'
f=open(path, 'rb')
data=pickle.load(f)
print(data)