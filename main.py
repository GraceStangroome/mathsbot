import matplotlib
import numpy as np
import scipy
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import string
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
Bayes Theorem (incl. joint probability)
Point Crossover

Upcoming is:
Binary to Decimal
Bit Swap
Calculating the fitness
    - Input a fitness function and some values and calculate the fitness
KNN
Linear Regression
    - This gives you the gradient, y-intercept, SSE (Sum Squared Error) and R Squared
Naive Bayes Classifier 
    - Calculates which class a new data point will be in
    

Type END to quit.
"""

print("Starting up")


# Function provided by Matthew Nagy, the CS chad
def getPossibilities(inList):
    numSize = 2 ** len(inList)
    result = [[(i >> j) & 0x1 > 0 for j in range(len(inList))] for i in range(numSize)]
    return result


def clean(setToClean):
    result = set().union(*setToClean)
    # Get it so only the events are in events (naturally)
    result.discard(")")
    result.discard("(")
    result.discard(",")
    result.discard("|")
    return result


def removeStuff(items, toRemove):
    for item in items.copy():  # because we're deleting things
        for thing in toRemove:
            if item == thing:
                items.discard(item)  # we just need to not iterate through its options for now
    return items


def calculateProbabilities(tofind, found):
    additions = []
    partialResult = 1
    possibilities = getPossibilities(tofind)
    for possibility in possibilities:
        for part in equation:
            partialResult = partialResult * float(input("What is the probability of P{0} given {1} is {2} and {3} "
                                                        .format(part, tofind, possibility, found)))
        additions.append(partialResult)
    result = np.sum(additions)
    return result

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
    user = input("What do you need to calculate? ").lower().strip()
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
            "What do you need to calculate? Prosterior (p), Likelihood (l), prior (r), Marginal (m) ").lower().strip()
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
                events = clean(equation)
                setting = set(input(
                    "What is the posterior you want to calculate? e.g. (W|S)"))
                thingsToSet = list(clean(setting))
                toSetNumerator = thingsToSet[0] + " = 1 and " + thingsToSet[1] + " = 1"
                toSetDenominator = thingsToSet[1] + " = 1"
                numeratEvents = removeStuff(events, thingsToSet)
                denomEvents = removeStuff(events, thingsToSet[1])
                # Now, events contains the things we need to run through
                numerator = calculateProbabilities(numeratEvents, toSetNumerator)
                denominator = calculateProbabilities(denomEvents, toSetDenominator)
                result = numerator / denominator
                print("Result = ", result)
            else:
                print("Sorry, I didn't understand, please try again.")
        if additional == "l":
            posterior = float(input("Please now type the posterior: "))
            prior = float(input("prior: "))
            marginal = float(input("Marginal: "))
            answer = (marginal * posterior) / prior
            print("The likelihood is: ", answer)
        if additional == "m":
            posterior = float(input("Please now type the posterior: "))
            prior = float(input("prior: "))
            likelihood = float(input("likelihood: "))
            answer = (likelihood * prior) / posterior
            print("The marginal is: ", answer)
        if additional == "r":
            posterior = float(input("Please now type the posterior: "))
            marginal = float(input("prior: "))
            likelihood = float(input("likelihood: "))
            answer = (marginal * posterior) / likelihood
            print("The prior is: ", answer)
        else:
            print("Sorry, I didn't understand.")
    elif user == "point crossover":
        print("You have selected point crossover for evolutionary algorithms")
        fault = True
        while fault:
            parentA = list(input("Please now enter parent A (or the first parent): "))
            parentB = list(input("Please now enter parent B (or the second parent): "))
            numOfPoints = int(input("How many point crossovers are you doing e.g. '2': "))
            if len(parentA) != len(parentB):  # User error catching
                print("Parent A and B MUST be the same length. Please try again.")
            else:
                fault = False
        points = []
        # Thanks to code from https://www.geeksforgeeks.org/python-single-point-crossover-in-genetic-algorithm/
        for i in range(numOfPoints):
            point = int(input("Please now enter point {0}: ".format(i)).strip())
            points.append(point)
            for gene in range(points[i], len(parentA)):  # parent A and B should be the same length
                parentA[gene], parentB[gene] = parentB[gene], parentA[gene]
        parentA = ''.join(parentA)
        parentB = ''.join(parentB)
        print("Child of A is: ", parentA)
        print("Child of B is: ", parentB)
    else:
        print(
            "Sorry, I didn't understand. I cannot interpret spelling mistakes, including extra spaces. "
            "Capitalisation doesn't matter.")
        print("I can currently calculate the following: ", possible)
