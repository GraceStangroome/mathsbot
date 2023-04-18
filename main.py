import matplotlib
import numpy as np
import scipy
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import string
from itertools import combinations
import matplotlib.pyplot as plt

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
Gaussian Probability (Normal Distribution)
    - Will find the probability density at a particular point given a list of numbers
Gaussian Mixture Model
    - estimates parameters of a Gaussian mixture model probability distribution
Bayes Theorem


Upcoming is:
KNN
Linear Regression
    - This gives you the gradient, y-intercept, SSE (Sum Squared Error) and R Squared
Naive Bayes Classifier 
    - Calculates which class a new data point will be in


Type END to quit.
"""

print("Starting up")

# Code provided by Matthew Nagy, the CS chad
def getPossibilities(inList):
    numSize = 2 ** len(inList)
    possibilities = [[(i >> j) & 0x1 > 0 for j in range(len(inList))] for i in range(numSize)]
    return possibilities

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
        elif what == "array":
            global arr
            arr = xvals
            print("Saved")
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


def getarray():
    global arr
    if usesaved():
        print("Using saved")
        return arr
    else:
        xvals = toarray(input("Enter all values, separated with a comma e.g. '2,4,6'"))
        savevalues(xvals, 0, "array")
    return xvals


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
arr = np.array([])
pointA = (0, 0)
pointB = (0, 0)
running = 0
while running == 0:
    user = input("What do you need to calculate?").lower().strip()
    if user == "end":
        print("You have chosen to end the program. Goodbye.")
        running = 1
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
        for i in range(len(xs.shape[0]) - 1):
            distance += (xs[i] - ys[i]) ** 2
        answer = sqrt(distance)
        print("Answer: ", answer)
    elif user == "sd":
        print("You have chosen standard deviation")
        array = getarray()
        answer = np.std(array)
        print("Answer: ", answer)
    elif user == "mean":
        print("You have chosen mean")
        array = getarray()
        answer = np.mean(array)
        print("Answer: ", answer)
    # thanks to https://datagy.io/python-euclidian-distance/
    elif user == "euclidean distance" or user == "euclidean":
        print("You have chosen Euclidean Distance")
        x, y = getpoints()
        squaredDistance = np.sum(np.square(x - y))
        answer = np.sqrt(squaredDistance)
        print("Answer: ", answer)
    # thanks to https://math.stackexchange.com/questions/139600/how-do-i-calculate-euclidean-and-manhattan-distance-by-hand
    elif user == "manhattan distance" or user == "manhattan":
        print("You have chosen Manhattan Distance")
        x, y = getpoints()
        distance = abs(x[0] - y[0]) + abs(x[1] - y[1])
        print("Answer: ", distance)
    # thanks to https://stackoverflow.com/questions/12412895/how-to-calculate-probability-in-a-normal-distribution-given-mean-standard-devi
    elif user == "gaussian probability" or user == "gaussian":
        print("You have chosen Gaussian probability")
        value = int(input("What value would you like the probability density to be caluclated at?: "))
        array = getarray()
        mean = np.mean(array)
        sd = np.std(array)
        answer = scipy.stats.norm(mean, sd).pdf(value)
        print("Probability Density: ", answer)
    elif user == "gaussian mixture model" or user == "mixture":
        print("You have chosen Gaussian Mixture Model")
        print("Please enter the TRAINING data")
        xs, ys = getxys()
        data = pd.DataFrame({'xs': xs, 'ys': ys})
        gmm = GaussianMixture(n_components=3)
        gmm.fit(data)
        plt.figure()
        plt.scatter(xs, ys)
        print("Weights: ", gmm.weights_)
        means = gmm.means_
        print("Means: ", means)
        print("Co-variances: ", gmm.covariances_)
        # thanks to https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
        for i, (X, Y) in enumerate(means):
            plt.scatter(X, Y, color='pink')
            # c=kmm.labels_, cmap='cool'
        plt.show()
        more = input("Would you like to predict labels for more data? Y/N ").lower().strip()
        if more == "y":
            print("Please enter the TESTING data")
            testXs, testYs = getxys()
            data = pd.DataFrame({'xs': testXs, 'ys': testYs})
            results = gmm.predict(data)
            print("Labels: ", results)
            plt.show()
    elif user == "bayes":
        additional = input(
            "What do you need to calculate? Prosterior (p), Likelihood (l), prior (p), Marginal (m) ").lower().strip()
        if additional == "p":
            theorem = input("Do you know the likelihood, Prior and Marginal? Y/N ").lower().strip()
            if theorem == "y":
                likelihood = float(input("Please now type the likelihood: "))
                prior = float(input("prior: "))
                marginal = float(input("Marginal: "))
                answer = (likelihood * prior) / marginal
                print("The prosterior is: ", answer)
            elif theorem == "n":
                equation = input("Please write the joint equation in full: ").strip()
                equation = equation.split("P")
                equation.pop(0)  # For some reason it always had an empty element in the start, so let's get rid of that
                events = set().union(*equation)
                # Get it so only the events are in events (naturally)
                events.discard(")")
                events.discard("(")
                events.discard(",")
                events.discard("|")
                thingsToSet = list()
                setting = input(
                    "Would you like to set any particular events e.g. P(W=1), If yes, type 'W=1' or just say 'N' ") # TODO: Add something here that instead gets the posterior the user is looking to calculate, and therefore auto calculates this instead
                if setting != "n":
                    thingsToSet = list(set().union(*setting))
                    for event in events.copy():  # because we're deleting things
                        for thing in thingsToSet:
                            if event == thing:
                                events.discard(event)  # we just need to not iterate through its options for now
                # Now, events contains the things we need to run through
                additions = []
                partialResult = 1
                possibilities = getPossibilities(events)
                for possibility in possibilities:
                    for part in equation:
                        partialResult = partialResult * float(input("What is the probability of P{0} given {1} is {2} and {3}"
                                                                    .format(part, events, possibility, thingsToSet)))
                    additions.append(partialResult)
                firstPart = np.sum(additions)
                print(firstPart)
            else:
                print("Sorry, I didn't understand, please try again.")
    else:
        print(
            "Sorry, I didn't understand. I cannot interpret spelling mistakes, including extra spaces. "
            "Capitalisation doesn't matter.")
        print("I can currently calculate the following: ", possible)
