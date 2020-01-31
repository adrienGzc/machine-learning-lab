import math
import numpy as np
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

def knn(trainData=None, targetData=None, testData=None, targetTest=None, kValue=3, labels=None, mode=classification):
  allDistances = np.array([])

  for testInstance in testData:
    eucDistances = np.array([])
    for trainInstance in trainData:
      eucDistances = np.append(eucDistances, [getDistance(testInstance, trainInstance)], axis=0)
    eucDistances = sorted(eucDistances)[:kValue]
    print(eucDistances)
    if allDistances.size == 0:
      np.insert(allDistances, 0, eucDistances, axis=0)
    else:
      allDistances = np.append(allDistances, eucDistances)
    print('all', allDistances)

  # k_nearest_labels = [trainData[i][1] for distance, i in kDistances]

  # return kDistances , mode(k_nearest_labels)

def main():
  # Get the iris data from sklearn
  irisData = datasets.load_iris()
  # Delete 1 type of flower in the data and in the target
  irisData.data = irisData.data[:-50]
  irisData.target = irisData.target[:-50]
  labels = irisData.target_names[:2]

  X_train, X_test, y_train, y_test = train_test_split(irisData.data, irisData.target, test_size=0.2, random_state=0)
  knn(trainData=X_train, targetData=y_train, testData=X_test, targetTest=y_test, kValue=3, labels=labels, mode=classification)
  # print(neighbors, predict)

if __name__ == "__main__":
    main()