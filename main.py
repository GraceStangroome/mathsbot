import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
# A python program that asks what you would like to do, and can calculate:
possible = """
Accuracy
Precision
Recall
MSE (Mean Squared Error)
MAE (Mean Absolute Error)
F1
KNN Weights (K Nearest Neighbour Weights)
SD (Standard Deviation)
Mean
Euclidean Distance
Manhattan Distance

Upcoming is:
Gaussian Probability
Bayes Theorem
KNN
Linear Regression
    - This gives you the gradient, y-intercept, SSE (Sum Squared Error) and R Squared
Gaussian Mixture Distribution
Naive Bayes Classifier 
    - Calculates which class a new data point will be in


Type END to quit.
"""


print("Starting up")


def usesaved():
    if globalsExist:
        use = input("Would you like to use your saved values? Y/N")
        if use == "Y" or use == "y":
            return True
    return False


def savevalues(xvals, yvals, what):
    global globalsExist
    save = input("Would you like to save these arrays and use them again? Y/N")
    if save == "Y" or save == "y":
        print("You chose to save the arrays. Saving...")
        globalsExist = True
        if what == "arrays":
            global globalxs
            global globalys
            globalxs = xvals
            globalys = yvals
            print("Saved")
            return True
        elif what == "points":
            global pointA
            global pointB
            pointA = xvals
            pointB = yvals
            print("Saved")
            return True
        else:
            print("An internal error has occurred. Values not saved. Try again.")
            return False
    return False


def toarray(vals):
    return np.array(vals.strip().split(","), float)


def getxys():
    global globalxs
    global globalys
    if usesaved():
        print("Using saved")
        return globalxs, globalys
    else:
        xvals = toarray(input("Enter all X values, separated with a comma e.g. '2,4,6'"))
        yvals = toarray(input("Enter all Y values, separated with a comma e.g. '1,3,7'"))
        savevalues(xvals, yvals, "arrays")
    return xvals, yvals


def getpoints():
    global pointA
    global pointB
    if usesaved():
        print("Using saved")
        return pointA, pointB
    else:
        a = toarray(input("Enter co-ordinates of the first point e.g. '2,4'"))
        b = toarray(input("Enter co-ordinates of the second point e.g. '5,7'"))
        savevalues(a, b, "points")
    return a, b


globalsExist = False
globalxs = np.array([])
globalys = np.array([])
pointA = (0, 0)
pointB = (0, 0)
a = 0
while a == 0:
    user = input("What do you need to calculate?").lower().strip()
    if user == "end":
        print("You have chosen to end the program. Goodbye.")
        a = 1
    elif user == "accuracy":
        print("You have chosen accuracy")
        correct = int(input("What are the number of correctly classified data points?"))
        total = int(input("What are the total number of datapoints?"))
        answer = correct / total
        print("Answer: ", answer)
    elif user == "precision":
        truePos = int(input("Number of true positives"))
        falsePos = int(input("Number of false positives"))
        answer = truePos / (truePos + falsePos)
        print("Answer: ", answer)
    elif user == "recall":
        truePos = int(input("Number of true positives"))
        falseNeg = int(input("Number of false negatives"))
        answer = truePos / (truePos + falseNeg)
        print("Answer: ", answer)
    elif user == "mse":
        print("You have chosen Mean Squared Error")
        xs, ys = getxys()
        mse = mean_squared_error(xs, ys)
        print("Answer: ", mse)
    elif user == "mae":
        print("You have chosen Mean absolute Error")
        xs, ys = getxys()
        mse = mean_absolute_error(xs, ys)
        print("Answer: ", mse)
    elif user == "f1":
        print("You have chosen F1")
        truePos = int(input("Number of true positives"))
        falseNeg = int(input("Number of false negatives"))
        falsePos = int(input("Number of false positives"))
        recall = truePos / (truePos + falseNeg)
        precision = truePos / (truePos + falsePos)
        answer = 2 * ((precision * recall) / (precision + recall))
        print("Answer: ", answer)
    elif user == "knn weights":
        print("You have chosen KNN Weights")
        xs, ys = getxys()
        uniform = np.sum(ys) / ys.shape[0]
        distance = 0
        for i in range(len(xs.shape[0]) -1):
            distance += (xs[i] - ys[i]) ** 2
        answer = sqrt(distance)
        print("Answer: ", answer)
    elif user == "sd":
        print("You have chosen standard deviation")
        vals = input("Enter all values, separated with a comma e.g. '2,4,6'")
        array = toarray(vals)
        answer = np.std(array)
        print("Answer: ", answer)
    elif user == "mean":
        print("You have chosen mean")
        vals = input("Enter all values, separated with a comma e.g. '2,4,6'")
        array = toarray(vals)
        answer = np.mean(array)
        print("Answer: ", answer)
    # thanks to https://datagy.io/python-euclidian-distance/
    elif user == "euclidean distance":
        x, y = getpoints()
        squaredDistance = np.sum(np.square(x - y))
        answer = np.sqrt(squaredDistance)
        print("Answer: ", answer)
    # thanks to https://math.stackexchange.com/questions/139600/how-do-i-calculate-euclidean-and-manhattan-distance-by-hand
    elif user == "manhattan distance":
        x, y = getpoints()
        distance = abs(x[0] - y[0]) + abs(x[1] - y[1])
        print("Answer: ", distance)
    else:
        print("Sorry, I didn't understand. I cannot interpret spelling mistakes, including extra spaces. Capitalisation doesn't matter.")
        print("I can currently calculate the following: ", possible)
