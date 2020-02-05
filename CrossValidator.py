import random
import pprint
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

  def score(self):
    if (self.__checkNotEmptyAttributes() is False):
      return False

    self.__splitDatasetIntoKFolds()
    # scores = list()
    for index, fold in enumerate(self.folds):
      trainData = self.folds.copy()
      testData = fold
      trainData.pop(index)
      trainData = sum(trainData, [])

      pprint.pprint(trainData)