import math
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

class KNearestNeighbor:
  def __init__(self, trainData=None, trainTarget=None, kValue=5, mode='classification'):
    self.trainData = trainData
    self.trainTarget = trainTarget
    self.kValue = kValue
    self.distances = [[]]
    self.predictionMode = mode

  def fit(self, data, targetData):
    self.trainData = data
    self.trainTarget = targetData
  
  def predict(self, labels):
    if self.predictionMode == 'classification':
      return Counter(labels).most_common(1)[0][0]
    elif self.predictionMode == 'regression':
      return sum(labels) / len(labels)
    else:
      return None

  def score(self, testData, testTarget):
    for i, testInstance in enumerate(testData):
      for _index, trainInstance in enumerate(self.trainData):
        eucDistance = self.getDistance(testInstance, trainInstance)
        self.distances[i].append(eucDistance)

  def getDistance(self, point1, point2):
    distanceSum = 0
    for i in range(len(point1)):
        distanceSum += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(distanceSum)

def main():
  # Get the iris data from sklearn
  irisData = datasets.load_iris()
  # Delete 1 type of flower in the data and in the target
  irisData.data = irisData.data[:-50]
  irisData.target = irisData.target[:-50]

if __name__ == "__main__":
  main()