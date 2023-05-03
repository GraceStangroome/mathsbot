import math
import numpy as np
import scipy
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from math import sqrt
import numexpr
import matplotlib.pyplot as plt
import re

# A python program that asks what you would like to do, and can calculate:
possible = """
Outside of brackets is the command, inside is further context
Accuracy
Precision
Recall
MSE (Mean Squared Error)
MAE (Mean Absolute Error)
F1
Prior Probability
KNN Weights (K Nearest Neighbour Weights)
SD (Standard Deviation)
Mean
Euclidean Distance
Manhattan Distance
Gaussian Probability (Normal Distribution)
    - Will find the probability density at a particular point given a list of numbers
Gaussian Mixture Model
    - estimates parameters of a Gaussian mixture model probability distribution
Bayes (Theorem incl. joint probability)
Point Crossover
Swap Mutation (bit swap)
Binary Conversion (binary to decimal)
Decimal Conversion (decimal to binary)
Fitness (of a function)
Linear Regression
    - (This gives you the gradient, y-intercept, SSE (Sum Squared Error) and R Squared)
Naive Bayes Classifier 
    - (Calculates which class a new data point will be in)
Clustering (via K-Means)

Upcoming is:
Value Iteration

Type END to quit.
"""

print("Starting up")


# Function provided by Matthew Nagy, the CS chad
def getPossibilities(inList):
    numSize = 2 ** len(inList)
    result = [[(i >> j) & 0x1 > 0 for j in range(len(inList))] for i in range(numSize)]
    return result


def clean(setToClean, unionise):
    if unionise:
        result = set().union(*setToClean)
        result.discard(")")
        result.discard("(")
        result.discard(",")
        result.discard("|")
    else:
        # Get it so only the events are in events (naturally)
        result = re.split("\[|\]|\)|'|\(|\{|\}| ", setToClean)
    return result


def removeStuff(items, toRemove):
    resultingItems = items.copy()
    for item in items:  # because we're deleting things
        for thing in toRemove:
            if item == thing:
                resultingItems.discard(item)  # we just need to not iterate through its options for now
    return resultingItems


def calculateProbabilities(equation, tofind, found):
    additions = []
    partialResult = 1
    possibilities = getPossibilities(tofind)
    for possibility in possibilities:
        for part in equation:
            if not tofind:
                # then toFind is empty
                partialResult = partialResult * float(input("What is P({0}) if we know {1} "
                                                            .format(part, found)))
            else:
                partialResult = partialResult * float(input("What is P({0}) if we know {1} is {2} and {3} "
                                                            .format(part, tofind, possibility, found)))
        additions.append(partialResult)
        partialResult = 1  # reset it for the next iteration
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
    save = input("Would you like to save these arrays and use them again? Y/N").lower()
    if save == "y":
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


def getParents():
    fault = True
    while fault:
        a = list(input("Please now enter parent A (or the first parent): "))
        b = list(input("Please now enter parent B (or the second parent): "))
        if len(a) != len(b):  # User error catching
            print("Parent A and B MUST be the same length. Please try again.")
        else:
            fault = False
    return a, b


def getPoints():
    numOfPoints = int(input("How many points are you considering e.g. '2': "))
    thePoints = []
    for num in range(numOfPoints):
        point = int(input("Please now enter point {0}: ".format(num)).strip())
        thePoints.append(point)
    return thePoints


def getValues():
    numOfValues = int(input("How many solutions are you calculating e.g. '5': "))
    theValues = []
    for num in range(numOfValues):
        point = float(input("Please now enter solution {0}: ".format(num)).strip())
        theValues.append(point)
    return theValues


# x here is a global variable now, that numexpr needs
def func(expr, x, a):
    return numexpr.evaluate(expr)


def least_squares_formula(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


def linear_resizer(x):
    return np.column_stack((np.ones(x.shape), x))


def linear_line(xs, ys):
    coefficients = least_squares_formula(linear_resizer(xs), ys)
    resulting_x = np.linspace(xs[0], xs[len(xs) - 1], len(xs))
    y_intercept, gradient = coefficients[0], coefficients[1]
    calculated_ys = y_intercept + gradient * resulting_x
    return resulting_x, calculated_ys, gradient, y_intercept


def squared_error(estimated_y, test_y):
    this_error = 0
    for i in range(len(estimated_y)):
        squared_difference = (estimated_y[i] - test_y[i]) ** 2
        this_error += squared_difference
    return this_error


def find_linear_error(gradient, y_intercept, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append(gradient * x + y_intercept)
    return squared_error(y_estimates, y_test)


# Thanks to https://stackoverflow.com/questions/20167108/how-to-check-how-many-times-an-element-exists-in-a-list
def marginalise(eqn, askingFor):
    cleaned = set(clean(eqn, False))
    # Didn't want to split on commas before,  but cleaned adds commas on their own
    # so lets clean the garbage
    cleaned.discard(",")
    cleaned.discard('')
    # continuing...
    conditionals = []
    notConditionals = []
    onLeft = []
    onRight = []
    for item in cleaned:
        stringVers = str(item)
        if "|" in stringVers:
            conditionals.append(item)
            splited = stringVers.split("|")
            onLeft.append(splited[0])
            onRight.append(splited[1])
            # because of this, onLeft[i] is the left part of conditionals[i]
            # and onRight[i] is the right part of conditionals[i]
        else:
            notConditionals.append(item)
    i = 0
    for item in onLeft:
        if item not in onRight:
            if item not in notConditionals:
                if item not in askingFor:
                    cleaned.remove(conditionals[i])
        i += 1
    return cleaned


def getpriorprob(output, size):
    if output:
        print("You have selected the prior probability")
    thisClass = int(input("How many points are in this class?: "))
    if size == 0:
        total = int(input("How many points are in the whole data set?: "))
    else:
        total = size
    result = thisClass / total
    if output:
        print("The prior probability is: ", result)
    return result


def valueIteration(gridWorld, gridWorldDimensions, cell, discount, probability):
    # coordinates
    cellRow = cell[0]
    cellColumn = cell[1]
    cellVal = gridWorld[cellRow][cellColumn]
    surroundingVals = []
    directions = []
    if cellRow + 1 >= gridWorldDimensions[0]:
        surroundingVals.append(cellVal)
        directions.append("down")
    if cellRow + 1 < gridWorldDimensions[0]:
        surroundingVals.append(gridWorld[cellRow + 1][cellColumn])
        directions.append("down")
    if gridWorldDimensions[0] > cellRow - 1 >= 0:  # because we do something else if it is less than 0
        surroundingVals.append(gridWorld[cellRow - 1][cellColumn])
        directions.append("up")
    if cellRow - 1 < 0:
        surroundingVals.append(cellVal)
        directions.append("up")
    if cellColumn + 1 >= gridWorldDimensions[1]:
        surroundingVals.append(cellVal)
        directions.append("right")
    if cellColumn + 1 < gridWorldDimensions[1]:
        surroundingVals.append(gridWorld[cellRow][cellColumn + 1])
        directions.append("right")
    if gridWorldDimensions[1] > cellColumn - 1 >= 0:
        surroundingVals.append(gridWorld[cellRow][cellColumn - 1])
        directions.append("left")
    if cellColumn - 1 < 0:
        surroundingVals.append(cellVal)
        directions.append("left")
    expectedBestIndex = surroundingVals.index(max(surroundingVals))
    expectedBest = surroundingVals[expectedBestIndex]
    direction = directions[expectedBestIndex]
    print("Expected Best: ", expectedBest)
    if len(set(surroundingVals)) == 1:  # sets remove duplicate values, so if everything is the same
        surroundingVals.remove(cellVal)   # all actions have the same utility
    del surroundingVals[expectedBestIndex]  # it was the best one, so it's not a value to consider anymore
    # Because we go in right angles to direction we go in to get best outcome, we will always do
    remainingProb = round((1-probability) / 2, 1)  # otherwise it gives a silly answer
    # going in right angles means going left or right if going up or down
    # and up and down if going left or right
    print("Best action is ", direction)
    otherDirectionVal = 0
    if direction == "up" or direction == "down":
        if gridWorldDimensions[1] > cellColumn - 1 >= 0:  # left
            otherDirectionVal += gridWorld[cellRow][cellColumn - 1] * remainingProb
        if cellColumn - 1 < 0:  # left
            otherDirectionVal += cellVal * remainingProb
        if cellColumn + 1 >= gridWorldDimensions[1]:  # right
            otherDirectionVal += cellVal * remainingProb
        if cellColumn + 1 < gridWorldDimensions[1]:  # right
            otherDirectionVal += gridWorld[cellRow][cellColumn + 1] * remainingProb
    elif direction == "right" or direction == "left":
        if cellRow + 1 >= gridWorldDimensions[0]:  # down
            otherDirectionVal += cellVal * remainingProb
        if cellRow + 1 < gridWorldDimensions[0]:  # down
            otherDirectionVal += gridWorld[cellRow + 1][cellColumn] * remainingProb
        if gridWorldDimensions[0] > cellRow - 1 >= 0:  # up
            otherDirectionVal += gridWorld[cellRow - 1][cellColumn] * remainingProb
        if cellRow - 1 < 0:
            otherDirectionVal += cellVal * remainingProb  # up
    else:
        print("internal error at going in right angles")
    update = cellVal + discount * (probability * expectedBest + otherDirectionVal)
    return update


def doIteration(gridWorld, newGridWorld, iteration, rows, cols, dimensions, discount, prob):
    print("For iteration {0}: ".format(iteration))
    for i in range(rows):
        for j in range(cols):
            result = valueIteration(gridWorld, dimensions, [i, j], discount, prob)
            newGridWorld[i][j] = result
            print("Value of cell [{0},{1}] is now {2}.".format(i, j, result))
    return newGridWorld


# thanks to https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def main():
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
        elif user == "prior probability" or user == "prior":
            getpriorprob(True, 0)
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
        elif user == "gaussian probability" or user == "density":
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
                "Would you like to rearrange Bayes Theorem (b) or do Joint Probability stuff (j) ").lower().strip()
            if additional == "j":
                conditional = input(
                    "Do you want to calculate a conditional probability e.g. P(A|B): y/n ").lower().strip()
                if conditional == "n":
                    what = input("What is it that you want to calculate? e.g. H ")
                    condition = input("What does {0} rely on e.g what is X if {0}|X: ".format(what))
                    rawEquation = "P" + what + "|" + condition
                    if "," in condition:
                        conditions = condition.split(",")
                        for i in conditions:
                            rawEquation += "P" + i
                    else:
                        rawEquation += "P" + condition
                    equation = rawEquation.split("P")
                    equation.pop(0)
                    found = what + " = 1 "
                    if "," in condition:
                        result = calculateProbabilities(set(equation), conditions, found)  # equation, toFind, found
                    else:
                        result = calculateProbabilities(set(equation), condition, found)  # equation, toFind, found
                    print("P({0}) = {1}".format(what, result))
                elif conditional == "y":
                    equation = input("Please write the joint equation in full: ").strip()
                    equation = equation.split("P")
                    # For some reason it always had an empty element in the start, so let's get rid of that
                    equation.pop(0)
                    setting = input("What is it that you want to calculate? e.g. W|S ")
                    if "|" in setting:
                        splited = setting.split("|")
                    thingsToSet = list(clean(setting, True))
                    marginalised = marginalise(str(equation), thingsToSet)  # keep this as is
                    # left side needs to be included in the numerator, but not in the denominator
                    toSetNumerator = ""
                    for i in thingsToSet:
                        toSetNumerator += i + " = 1 "
                    # denominator is the !first one
                    denomThingsToSet = thingsToSet.copy()
                    denomThingsToSet.remove(splited[0])
                    toSetDenominator = ""
                    for i in denomThingsToSet:
                        toSetDenominator += i + " = 1 "
                    events = clean(marginalised, True)
                    numeratEvents = removeStuff(events, thingsToSet)
                    denomEvents = removeStuff(events, denomThingsToSet)
                    # Now, events contains the things we need to run through
                    numerator = calculateProbabilities(events, numeratEvents, toSetNumerator)
                    denominator = calculateProbabilities(events, denomEvents, toSetDenominator)
                    result = numerator / denominator
                    print("Result = ", result)
                else:
                    print("Sorry, I didn't understand, please try again.")
            elif additional == "b":
                theorem = input(
                    "What do you need to calculate? Posterior (p), Likelihood (l), prior (r), Marginal (m) ") \
                    .lower().strip()
                if theorem == "p":
                    likelihood = float(input("Please now type the likelihood: "))
                    prior = float(input("prior: "))
                    marginal = float(input("Marginal: "))
                    answer = (likelihood * prior) / marginal
                    print("The prosterior is: ", answer)
                elif theorem == "l":
                    posterior = float(input("Please now type the posterior: "))
                    prior = float(input("prior: "))
                    marginal = float(input("Marginal: "))
                    answer = (marginal * posterior) / prior
                    print("The likelihood is: ", answer)
                elif theorem == "m":
                    posterior = float(input("Please now type the posterior: "))
                    prior = float(input("prior: "))
                    likelihood = float(input("likelihood: "))
                    answer = (likelihood * prior) / posterior
                    print("The marginal is: ", answer)
                elif theorem == "r":
                    posterior = float(input("Please now type the posterior: "))
                    marginal = float(input("prior: "))
                    likelihood = float(input("likelihood: "))
                    answer = (marginal * posterior) / likelihood
                    print("The prior is: ", answer)
                else:
                    print("Sorry, I didn't understand.")
            else:
                print("Sorry, I didn't understand.")
        elif user == "point crossover":
            print("You have selected point crossover for evolutionary algorithms")
            parentA, parentB = getParents()
            points = getPoints()
            # Thanks to code from https://www.geeksforgeeks.org/python-single-point-crossover-in-genetic-algorithm/
            for i in range(len(points)):
                for gene in range(points[i], len(parentA)):  # parent A and B should be the same length
                    parentA[gene], parentB[gene] = parentB[gene], parentA[gene]
            parentA = ''.join(parentA)  # Makes it look like a string in the output
            parentB = ''.join(parentB)
            print("Child of A is: ", parentA)
            print("Child of B is: ", parentB)
        elif user == "swap mutation":
            print("You have selected swap crossover for evolutionary algorithms")
            print("Note: This only works for an even number of swap locations.")
            parent = list(input("Please now enter parent A (or the first parent): "))
            points = getPoints()
            for i in range(len(points) - 1):
                parent[points[i]], parent[points[i + 1]] = parent[points[i + 1]], parent[points[i]]
            parentRes = ''.join(parent)  # Makes it look like a string in the output
            print("Child of parent is now: ", parentRes)
        elif user == "binary conversion" or user == "binary":
            print("You have selected binary to decimal conversion")
            binary = input("Please enter your binary number: ")
            # https://pythonguides.com/python-convert-binary-to-decimal/
            result = int(binary, 2)
            print("The Decimal value is: ", result)
        elif user == "decimal conversion" or user == "decimal":
            print("You have selected decimal to binary conversion")
            decimal = int(input("Please enter your decimal number: "))
            # https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
            result = '{0:08b}'.format(decimal)
            print("The binary value is: ", result)
        elif user == "fitness":
            print("You have selected fitness proportionate selection")
            # thank you to https://stackoverflow.com/questions/52596765/how-to-get-equation-from-input-in-python !!
            rawEqn = input("Please enter the fitness equation: ")
            inputs = getValues()
            rawResults = []
            needToRescale = False
            for i in range(len(inputs)):
                rawResult = func(rawEqn, inputs[i], 0)
                rawResults.append(float(rawResult))
                if rawResult < 0:
                    needToRescale = True
            print("Raw Results: ", rawResults)
            using = rawResults
            if needToRescale:
                print("Rescaling...")
                rescaledResults = []
                smallestElem = min(rawResults) * -1
                rescaleEqn = "x + a"
                for i in range(len(rawResults)):
                    rescaled = func(rescaleEqn, rawResults[i], smallestElem)
                    rescaledResults.append(float(rescaled))
                print("Rescaled Values: ", rescaledResults)
                using = rescaledResults
            else:
                print("No need to rescale.")
            total = sum(using)
            relativeFitness = []
            for i in range(len(using)):
                result = using[i] / total
                relativeFitness.append(result)
            print("Relative Fitness: ", relativeFitness)
            ranges = []
            tops = [0]
            for i in range(len(relativeFitness)):
                if i == 0:
                    a = 0
                else:
                    if relativeFitness[i - 1] != 0:
                        a = tops[i]
                    else:
                        a = tops[i - 1]
                top = a + relativeFitness[i]
                if top == a:
                    ranges.append(a)
                else:
                    ranges.append(str(a) + " to " + str(top))
                tops.append(top)
            print("Ranges: ", ranges)
        elif user == "naive bayes classifier" or user == "naive":
            print("You have selected gaussian naive bayes")
            # Basically doing this: L = log(P(c)) + log(P(x1 | c)) + log(P(x2 | c))
            # Log Likelihood
            numClasses = int(input("How many classes are there: "))
            rawVariables = input("What are your variables? Separate with a comma only e.g. Density,Hardness ").strip()
            newPoint = toarray(input("What is the new point (x1,x2) you are trying to classify e.g. 5,3 "))
            knowingprior = input("Do you already know the prior probabilities? y/n ").lower().strip()
            if knowingprior == "n":  # I want to get from the users now about how many things are in the whole data set
                # so that we don't have to ask them loads of times which might be confusing
                dataSize = int(input("How many data points are there across all classes? "))
            priors = []
            knowingMeans = input(
                "Do you already know the mean and standard deviations for each class? y/n ").lower().strip()
            variables = rawVariables.split(",")
            likelihoods = []
            for i in range(numClasses):
                part = []  # reset it every time we get to a new class
                if knowingprior == "y":
                    prior = float(input("Please enter the prior probability for class {0} as a decimal ".format(i)))
                    priors.append(prior)
                else:
                    print("For class ", i)
                    tellUser = input("Would you like to know the prior probability? y/n ").lower().strip()
                    if tellUser == "y":
                        prior = getpriorprob(True, dataSize)
                    else:
                        prior = getpriorprob(False, dataSize)
                    priors.append(prior)
                for index, var in enumerate(variables):
                    if knowingMeans == "y":
                        mean = float(input("Please enter the Mean for {0} in class {1} as a decimal ".format(var, i)))
                        std = float(input(
                            "Please enter the Standard Deviation for {0} in class {1} as a decimal ".format(var, i)))
                    else:
                        message = "Enter all " + var + " values in class " + str(
                            i) + " separated with a comma e.g. '2,4,6 "
                        values = toarray(input(message))
                        mean = np.mean(values)
                        std = np.std(values)
                    firstHalf = (1 / (std * sqrt(2 * math.pi)))
                    b = -0.5 * ((newPoint[index] - mean) / std) ** 2
                    secondHalf = math.exp(b)
                    if secondHalf == 0:
                        # Wolfram Alpha can recognise really really tiny numbers as not 0 (like 10^-431),
                        # There’s only 10^80 atoms in the universe or something, so I'm really impressed with WA
                        # But calculators and python cannot handle this, they just evaluate it to 0
                        # We can do the following because log[A*exp(B)] = logA + log(expB) = logA + B
                        part.append(math.log(firstHalf) + b)
                    elif firstHalf == 0:
                        # we can't interchange log(A) + B
                        # so there has probably been a huge mistake
                        print("WARNING: I've detected an undefined maths function.")
                        print("I recommend that you start again, as I think a mistake has probably been made.")
                        print("Additionally, you may want to do this on Wolfram Alpha if this keeps happening.")
                        print("I can try and carry on though, which I will, and assume the problematic value is 0.")
                        print("I.e. this means we just pretend it didn't happen.")
                        part.append(0)  # why not lol
                    else:
                        probability = math.log(firstHalf * secondHalf)
                        part.append(probability)
                result = math.log(priors[i]) + part[0] + part[1]
                likelihoods.append(result)
                print("The log likelihood for this class is: ", result)
            maxedL = likelihoods.index(max(likelihoods))
            print("The class to choose would be {0} with a value of {1}".format(maxedL, likelihoods[maxedL]))
        elif user == "linear regression" or user == "linear":
            print("You have selected linear regression")
            fig, ax = plt.subplots()
            xs, ys = getxys()
            ax.scatter(xs, ys, label='Data', color="midnightblue")
            plt.xlabel('x')
            plt.ylabel('y')
            # Doing linear algebra with my old DDCS code that I think works
            linear_xs, linear_ys, gradient, y_intercept = linear_line(xs, ys)
            print("Gradient: ", gradient)
            print("Y Intercept: ", y_intercept)
            sse = find_linear_error(gradient, y_intercept, xs, ys)
            print("Sum Squared Error: ", sse)
            print("R Squared: ", r2_score(linear_ys, ys))
            ax.plot(linear_xs, linear_ys, color='pink', label='predicted values')
            plt.show()
        elif user == "value" or user == "value iteration":
            print("You have chosen value iteration for Markov Decision Processes")
            numRows = int(input("How many rows are there in the grid world: "))
            numColumns = int(input("How many columns are there in the grid world: "))
            dimensions = [numRows, numColumns]
            gridWorld = []
            i = 0
            while i < numRows:
                row = toarray(input("Please enter row {0} separated with commas only like 5,4,3: ".format(i)))
                if len(row) != numColumns:
                    print("Row entered had incorrect number of Columns, please try again.")
                else:
                    gridWorld.append(row)
                    i += 1  # move on
            discountFactor = float(input("What is the discount factor: "))
            message = "What is the probability that the transition model will go in the intended direction?: "
            intendedDirectionProb = float(input(message))
            numIterations = int(input("How many iterations would you like to do?: "))
            newGridWorld = [[0 for x in range(numColumns)] for y in range(numRows)]
            for iteration in range(numIterations):
                # update gridWorld
                gridWorld = doIteration(gridWorld, newGridWorld, iteration, numRows, numColumns, dimensions, discountFactor, intendedDirectionProb)
        elif user == "clustering":
            print("You have selected Clustering via K-Means")
            xs, ys = getxys()
            numCentroids = int(input("How many initialised centroids (0 if none): "))  # this is k
            centroids = []
            if numCentroids > 0:
                iterations = int(input("How many iterations would you like to do?: "))
                for iteration in range(iterations):
                    for i in range(numCentroids):
                        centroid = toarray(input("Please now enter centroid {0}: ".format(i)))
                        centroids.append(centroid)
                    closestCentroids = []
                    for index, x in enumerate(xs):
                        distances = []
                        point = toarray(str(x) + "," + str(ys[index]))
                        for centroid in centroids:
                            squaredDistance = np.sum(np.square(centroid - point))
                            distances.append(squaredDistance)
                            # print("distance between {0} and {1} is {2}".format(centroid, point, squaredDistance))
                        bestCentroidIndex = distances.index(min(distances))
                        closestCentroids.append(bestCentroidIndex)
                        print("Closest Centroid to point {0} is {1} ".format(point, centroids[bestCentroidIndex]))
                    plt.scatter(xs, ys, c=closestCentroids, cmap='cool')
                    plt.show()
                    # create new centroids from the means of our classification
                    for i in range(len(centroids)):
                        # this gets the indexes of all the points with the same centroid
                        groupI = list_duplicates_of(closestCentroids, i)
                        groupIXs = []
                        groupIYs = []
                        for member in groupI:
                            groupIXs.append(xs[member])
                            groupIYs.append(ys[member])
                        newGroupIX = np.mean(np.array(groupIXs))
                        newGroupIY = np.mean(np.array(groupIYs))
                        print("New Centroid is: ({0},{1})".format(newGroupIX, newGroupIY))
            else:
                X = linear_resizer(xs)
                kmm = KMeans()
                kmm.fit(X)
                plt.figure()
                print(kmm.labels_)
                plt.scatter(xs, ys, c=kmm.labels_, cmap='cool')
                plt.show()
        elif user == "help":
            print("I can currently calculate the following: ", possible)
        else:
            print(
                "Sorry, I didn't understand. I cannot interpret spelling mistakes, including extra spaces. "
                "Capitalisation doesn't matter.")
            print("I can currently calculate the following: ", possible)


globalsExist = False
globalxs = np.array([])
globalys = np.array([])
arr = np.array([])
pointA = (0, 0)
pointB = (0, 0)
main()
