import math
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def regression(labels):
  return sum(labels) / len(labels)

def classification(labels):
  return Counter(labels).most_common(1)[0][0]

def getDistance(instance, target):
  distanceSum = 0
  for i in range(len(instance)):
      distanceSum += math.pow(instance[i] - target[i], 2)
  return math.sqrt(distanceSum)

def knn(trainData=None, targets=None, kValue=3, mode='classification'):
  allDistances = []
  for index, instance in enumerate(trainData):
    eucDistance = getDistance(instance, targets)
    allDistances.append((eucDistance, index))

  sorted_neighbor_distances_and_indices = sorted(allDistances)

  k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:kValue]
  
  k_nearest_labels = [trainData[i][1] for distance, i in k_nearest_distances_and_indices]

  if mode == 'classification': mode = classification
  elif mode == 'regression': mode = regression

  return k_nearest_distances_and_indices , mode(k_nearest_labels)

def main():
  # Get the iris data from sklearn
  irisData = datasets.load_iris()
  # Delete 1 type of flower in the data and in the target
  irisData.data = irisData.data[:-50]
  irisData.target = irisData.target[:-50]

  X_train, X_test, y_train, y_test = train_test_split(irisData.data, irisData.target, test_size=0.2, random_state=0)
  neighbors, predict = knn(trainData=X_train, targets=X_test[0], mode='classification')
  print(neighbors, predict)

if __name__ == "__main__":
    main()