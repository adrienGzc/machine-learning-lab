import math
import pprint
from collections import Counter

class NaiveBayes:
  def __init__(self, trainData=None, trainTarget=None, testData=None, testTarget=None):
    self.splitedClasses = dict()
    self.trainedFeature = dict()

    self.trainData = trainData
    self.trainTarget = trainTarget
    self.testData = testData
    self.testTarget = testTarget


  def printSplitedClasses(self):
    pprint.pprint(self.splitedClasses)


  def testZip(self, data):
    for column in zip(*data):
      print(column)
    return self


  def __mean(self, feature):
    return (sum(feature) / len(feature))


  def __variance(self, feature):
    meanDifference = list()
    mean = self.__mean(feature)

    for instance in feature:
      meanDifference.append(pow((instance - mean), 2))
    return self.__mean(meanDifference)


  def __standardDeviation(self, feature):
    return round(math.sqrt(self.__variance(feature)), 2)


  def __classSpliter(self):
    for index, target in enumerate(self.trainData):
      if (self.trainTarget[index] not in self.splitedClasses):
        self.splitedClasses[self.trainTarget[index]] = list()
      self.splitedClasses[self.trainTarget[index]].append(target)
    return self.splitedClasses


  def setTrainData(self, trainData, targetData):
    self.trainData = trainData
    self.trainTarget = targetData


  def setTestData(self, testData, targetTest):
    self.testData = testData
    self.targetTest = targetTest


  def __squashFeature(self, classData):
    tmp = list()

    for feature in zip(*classData):
      tmp.append((len(feature), self.__mean(feature), self.__standardDeviation(feature)))
    return tmp


  def __gaussianPDF(self, feature, mean, standardDeviation):
    exponent = math.exp(-((feature - mean) ** 2 / (2 * standardDeviation ** 2 )))
    return (1 / (math.sqrt(2 * math.pi) * standardDeviation)) * exponent


  def fit(self, print=False):
    splitedCLasses = self.__classSpliter()

    for key, value in splitedCLasses.items():
      self.trainedFeature[key] = self.__squashFeature(value)    
    
    if (print == True):
      pprint.pprint(self.trainedFeature)
    return self


  def __getPropabilities(self, testInstance):
    probabilities = dict()

    for key, value in self.trainedFeature.items():
      probabilities[key] = value[0][0] / len(self.trainData)
      for index in range(len(value)):
        _nbInstance, mean, standardDeviation = value[index]
        probabilities[key] *= self.__gaussianPDF(testInstance[index], mean, standardDeviation)
    return probabilities


  def __getPredictionForInstance(self, testInstance):
    higherProb = None
    higherClass = None
    probaResults = self.__getPropabilities(testInstance)

    for key, value in probaResults.items():
      if (higherProb is None or higherProb < value):
        higherProb = value
        higherClass = key
    return higherClass

  def predict(self):
    predictions = list()

    for testInstance in self.testData:
      predictions.append(self.__getPredictionForInstance(testInstance))
    return predictions