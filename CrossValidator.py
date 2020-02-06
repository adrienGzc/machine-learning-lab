import random
import pprint
import copy
from functools import reduce

class CrossValidator:
  def __init__(self, algo=None, dataset=None, nbFolds=10):
    random.seed(1)
    self.folds = list()

    self.algorithm = algo
    self.dataset = list(dataset)
    self.nbFolds = nbFolds

  def __checkNotEmptyAttributes(self):
    if (self.algorithm is None or self.dataset is None or self.nbFolds <= 1):
      print("Error: Algorithm and dataset shouldn't be empty and nbFolds neither less nor equal to 0")
      return False
    return True

  def __getFoldSize(self, nbInstances, nbFolds):
    return round(nbInstances / nbFolds)

  def __splitDatasetIntoKFolds(self):
    copyDataset = self.dataset.copy()
    random.shuffle(copyDataset)
    foldSize = self.__getFoldSize(len(self.dataset), self.nbFolds)

    for nb in range(self.nbFolds):
      start = foldSize * nb
      end = foldSize + start
      self.folds.append(copyDataset[start:end])

  def __getTargetFromData(self, dataset):
    test = [instance.pop(-1) for instance in dataset]
    return test

  def __getAccuracy(self, original, predicted):
    nbCorrectPrediction = 0
    for index, predictClass in enumerate(predicted):
      if original[index] == predictClass:
        nbCorrectPrediction += 1
    return nbCorrectPrediction / len(original) * 100

  def score(self):
    if (self.__checkNotEmptyAttributes() is False):
      return False

    self.__splitDatasetIntoKFolds()
    accuracyScores = list()
    for index, fold in enumerate(self.folds):
      trainData = copy.deepcopy(self.folds)
      trainData.pop(index)
      trainData = sum(trainData, [])
      targetTrain = self.__getTargetFromData(trainData)
      testData = copy.deepcopy(fold)
      targetTest = self.__getTargetFromData(testData)


      self.algorithm.fit(trainData, targetTrain, fold, False)
      predictionFold = self.algorithm.predict()
      accuracyScores.append(self.__getAccuracy(targetTest, predictionFold))    
    return accuracyScores, sum(accuracyScores) / len(accuracyScores)