from sklearn.tree import DecisionTreeClassifier
features=[[140,0],[130,0],[150,1],[170,1]]
labels=[0,0,1,1]
clf=DecisionTreeClassifier()
clf.fit(features,labels)
p = clf.predict([[160,1]])
print("Prediction ",p)
