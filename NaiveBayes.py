import math
import pprint

class NaiveBayes:
  def __init__(self):
    self.splitedClasses = dict()
    self.trainedFeature = dict()

    self.trainData = None
    self.trainTarget = None
    self.testData = None


  # Return the mean of a list.
  def __mean(self, feature):
    return sum(feature) / len(feature)


  # Return the variance of a list.
  def __variance(self, feature):
    meanDifference = list()
    mean = self.__mean(feature)

    for instance in feature:
      meanDifference.append(pow((instance - mean), 2))
    return self.__mean(meanDifference)


  # Return the standard deviation from a list.
  def __standardDeviation(self, feature):
    return math.sqrt(self.__variance(feature))


  # Divide data by class.
  def __classSpliter(self):
    for index, target in enumerate(self.trainData):
      # Create new key in dict if class not already created.
      if (self.trainTarget[index] not in self.splitedClasses):
        self.splitedClasses[self.trainTarget[index]] = list()
      # Add the instance to the corresponding class.
      self.splitedClasses[self.trainTarget[index]].append(target)
    return self.splitedClasses


  # Store training data in the Naive Bayes class
  def __setTrainData(self, trainData, targetData):
    self.trainData = trainData
    self.trainTarget = targetData


  # Store test data in the Naive Bayes class
  def __setTestData(self, testData):
    self.testData = testData


  # Calcule mean and standard deviation for each column (feature) in the training data.
  def __squashFeature(self, classData):
    tmp = list()

    # The built-in zip give me all data from 1 column (feature), at once, on the classData list.
    for feature in zip(*classData):
      tmp.append((len(feature), self.__mean(feature), self.__standardDeviation(feature)))
    return tmp


  # Awful to read, I apologize for that, but correspond to the gaussian normal distribution.
  def __gaussian(self, feature, mean, standardDeviation):
    return (1 / (math.sqrt(2 * math.pi) * standardDeviation)) * math.exp(-((feature - mean) ** 2 / (2 * standardDeviation ** 2 )))

  # Return the classes probability based on the instance given.
  def __getPropabilities(self, testInstance):
    probabilities = dict()

    for key, value in self.trainedFeature.items():
      probabilities[key] = value[0][0] / len(self.trainData)

      for index in range(len(value)):
        # Get the mean and standard deviation from each feature
        _nbInstance, mean, standardDeviation = value[index]
        # Multiply the gaussian from each feature 
        probabilities[key] *= self.__gaussian(testInstance[index], mean, standardDeviation)
    return probabilities

  # Return, for the instances, the predicted class + the probabilities in a tuple.
  def __getPredictionForInstance(self, testInstance):
    classValue = None
    predictedClass = None
    # Get the class probability for the test instance.
    probaResults = self.__getPropabilities(testInstance)

    # Check which class as the most probability, this is gonna be out prediction.
    for key, value in probaResults.items():
      # If first loop lap OR the probability is higher than the actual then set the new class as the highest proba.
      if (classValue is None or classValue < value):
        classValue = value
        predictedClass = key
    return ((predictedClass, probaResults))


  # Training method. Take the training data and the target of the training data.
  def fit(self, trainData=None, trainTarget=None, displayTraining=False):

    # Store the data + label in the Naive Bayes class.
    self.__setTrainData(trainData, trainTarget)

    # Clean variables before to use it if algo used in cross validation or in a loop.
    self.trainedFeature.clear()
    self.splitedClasses.clear()

    # Return a dictionnary with all the different label as a key with the instances related in it.
    splitedCLasses = self.__classSpliter()

    # For every class, we reduce all the instance into the number of instances, the mean and the standard deviation.
    for key, value in splitedCLasses.items():
      self.trainedFeature[key] = self.__squashFeature(value)

    if (displayTraining == True):
      pprint.pprint(self.trainedFeature)
    return self


  # Method to predict, should be used after the fit method.
  def predict(self, testData):
 
    # If there is no training done before then return false with error message.
    if len(self.trainedFeature) is 0:
      print('Error: no training data recorded. Please fit (train) before to predict.')
      return False

    self.__setTestData(testData)

    # For every test instance we predict the class which she is the closer.
    predictions = list()
    for testInstance in self.testData:
      predictions.append(self.__getPredictionForInstance(testInstance))
    return predictions
