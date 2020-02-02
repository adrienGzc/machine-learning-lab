import math
import pprint
from collections import Counter

class NaiveBayes:
  def __init__(self, trainData, trainTarget, testData, testTarget):
    self.splitedClasses = dict()
    self.trainData = trainData
    self.trainTarget = trainTarget
    self.testData = testData
    self.testTarget = testTarget

  def printSplitedClasses(self):
    pprint.pprint(self.splitedClasses)

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

  def __classSplit(self):
    for index, target in enumerate(self.trainData):
      if self.trainTarget[index] not in self.splitedClasses:
        self.splitedClasses[self.trainTarget[index]] = list()
      self.splitedClasses[self.trainTarget[index]].append(target)
    return self

  def testZip(self, data):
    for column in zip(*data):
      print(column)
    return self

  def fit(self):
    return self

  def predict(self):
    return self