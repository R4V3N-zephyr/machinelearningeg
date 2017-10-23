from sklearn.tree import DecisionTreeClassifier
x=[[181,80,44],[177,70,43],[160,60,38],[158,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],[159,55,37],[171,75,42],[181,85,43]]
y=[1,1,0 ,0,1,1,0,0,0,1,1]
clf=DecisionTreeClassifier()
clf.fit(x,y)
sex=clf.predict([[190,70,43]])
if sex==1:
	print("Its a male")
else :
	print("Its a female")
