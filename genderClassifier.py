from sklearn import tree, svm, neighbors, ensemble 
import random
#Goal is to classify people based on 3 variables
#We store our 3 variables in a list [height, weight, shoe size]
X = []
Y = []

def convertShoeSize(shoe):
	return (((shoe+23.5)/3)*2.54*1.5)+1.5

def generateMale():
	return [random.normalvariate(176,7), random.normalvariate(89,12), random.normalvariate(convertShoeSize(10),convertShoeSize(1.468))]

def generateFemale():
	return [random.normalvariate(162,6), random.normalvariate(76,10), random.normalvariate(convertShoeSize(8),convertShoeSize(1.468))]

for i in range(5000):
	X.append(generateMale())
	Y.append('male')
	X.append(generateFemale())
	Y.append('female')

#We make classifiers fitted with our manually stored data
treeClassifier = tree.DecisionTreeClassifier().fit(X, Y)
svmClassifier = svm.SVC().fit(X, Y)
kNeighborsClassifier = neighbors.KNeighborsClassifier(n_neighbors=3).fit(X,Y)
bagClassifier = ensemble.BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5, max_features=0.5).fit(X,Y)

def singleGenderClassify(height, weight, shoeSize):
	#make our prediction variables and get the predictions of our input
	treePrediction = treeClassifier.predict([[height, weight, shoeSize]])
	svcPrediction = svmClassifier.predict([[height, weight, shoeSize]])
	kNeighborPrediction = kNeighborsClassifier.predict([[height, weight, shoeSize]])
	bagPrediction = bagClassifier.predict([[height, weight, shoeSize]])

	#Print out each model's prediction
	print("The decision tree model predicts: " + treePrediction[0])
	print("The support vector machine model predicts: " + svcPrediction[0])
	print("The k neigherest neighbor model predicts: " + kNeighborPrediction[0])
	print("The ensemble model predicts: " + bagPrediction[0])

def multipleGenderClassify(myList):
	#Calls singleGenderClassify on each person in myList
	for i in range(len(myList)):
		print("\nPerson " + str(i))
		singleGenderClassify(myList[i][0], myList[i][1], myList[i][2])

def customSingleClassify():
	#Get user input for their dimensions and convert to metric for the training data we used
	inputType = input("Please enter Y if you would like to use Metric or N if not\n")
	height = int(input("Please enter your height in cm\n")) if inputType == "Y" else int(input("Please enter your height in inches\n")) * 2.54
	weight = int(input("Please enter your weight in kg\n")) if inputType == "Y" else int(input("Please enter your weight in pounds\n"))/2.2046226218
	shoeSize = int(input("Please enter your shoe size in european sizing\n")) if inputType == "Y" else convertShoeSize(int(input("Please enter your shoe size (please use men's sizing)\n")))
	singleGenderClassify(height, weight, shoeSize)

def menu():
	print('-----------------------')
	print('|  Gender Classifier  |')
	print('-----------------------')
	print('|A| Enter Single Input|')
	print('|B| Generate Male     |')
	print('|C| Generate Female   |')
	print('|Q| Quit              |')
	answer = input('-----------------------\n')

	if(answer == "A" or answer == "a"):
		customSingleClassify()
	elif(answer == "B" or answer == "b"):
		person = generateMale()
		singleGenderClassify(person[0], person[1], person[2])
	elif(answer == "C" or answer == "c"):
		person = generateFemale()
		singleGenderClassify(person[0], person[1], person[2])
	elif(answer == "Q" or answer ==  "q"):
		exit()
	menu()
menu()