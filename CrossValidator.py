class CrossValidator:
  def __init__(self, algo=None, dataset=None, nbFolds=10):
    self.folds = list()

    self.algorithm = algo
    self.dataset = dataset
    self.nbFolds = nbFolds

  def __checkNotEmptyAttributes(self):
    if (self.algorithm is None or self.dataset is None or self.nbFolds <= 1):
      print("Error: Algorithm and dataset shouldn't be empty and nbFolds neither less nor equal to 0")
      return False
    return True

  def __splitDatasetIntoKFolds(self):
    foldSize = round(len(self.dataset) / self.nbFolds)
    
    for index in range(self.nbFolds):
      pass

  def score(self):
    if (self.__checkNotEmptyAttributes() is False):
      return False