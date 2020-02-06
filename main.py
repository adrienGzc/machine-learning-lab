import random
import pprint
from sklearn import datasets
from sklearn.model_selection import train_test_split

import NaiveBayes
import CrossValidator

def concateTargetWithDataset(dataset, targetDataset):
  data = list()
  for index, instance in enumerate(dataset):
    tmp = list(instance)
    tmp.append(targetDataset[index])
    data.append(tmp)
  return data

def main():
  irisData = datasets.load_iris()
  irisData.data = irisData.data[:-50]
  irisData.target = irisData.target[:-50]
  newDataset = concateTargetWithDataset(irisData.data, irisData.target)

  naiveBayes = NaiveBayes.NaiveBayes()
  crossValidator = CrossValidator.CrossValidator(algo=naiveBayes, dataset=newDataset, nbFolds=10)
  _scores, accuracy = crossValidator.score()
  predicton = naiveBayes.getPredictions()
  print(predicton)
  print('Accuracy: %.2f%%' % accuracy)

if __name__ == "__main__":
    main()