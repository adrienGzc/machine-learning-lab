import math
import numpy
import pprint
from sklearn import datasets
from collections import Counter
from sklearn.model_selection import train_test_split

class NaiveBayes:
  def __init__(self, trainData=None, trainTarget=None, testData=None, testTarget=None):
    self.allClasses = dict()
    self.trainData = trainData
    self.trainTarget = trainTarget
    self.testData = testData
    self.testTarget = testTarget

  def printAllClasses(self):
    pprint.pprint(self.allClasses)

  def checkEmptyData(self):
    all(self.trainData)
    all(self.trainTarget)
    all(self.testData)
    all(self.testTarget)

  def fit(self, trainData, trainTarget, testData, testTarget):
    self.trainData = trainData
    self.trainTarget = trainTarget
    self.testData = testData
    self.testTarget = testTarget
    return self

  def classSplit(self):
    if (self.checkEmptyData() is False):
      print('Error: cannot perform a split on empty data. ')
      return False

    for index, target in enumerate(self.trainData):
      if self.trainTarget[index] not in self.allClasses:
        self.allClasses[self.trainTarget[index]] = list()
      self.allClasses[self.trainTarget[index]].append(target)
    return self

  def gaussienDensity(self):
    pass

  def score(self):
    pass


def main():
  irisData = datasets.load_iris()
  nBayes = NaiveBayes()
  X_train, X_test, y_train, y_test = train_test_split(irisData.data, irisData.target, test_size=0.2, random_state=0)
  nBayes.fit(trainData=X_train, trainTarget=y_train, testData=X_test, testTarget=y_test).classSplit().printAllClasses()

if __name__ == "__main__":
    main()