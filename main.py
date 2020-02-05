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
  X_train, X_test, y_train, y_test = train_test_split(irisData.data, irisData.target, test_size=0.2, random_state=0)

  naiveBayes = NaiveBayes.NaiveBayes()
  crossValidator = CrossValidator.CrossValidator(algo=naiveBayes, dataset=newDataset, nbFolds=5)
  crossValidator.score()
  predict = naiveBayes.fit(trainData=X_train, trainTarget=y_train, testData=X_test, testTarget=y_test).predict()
  # print(predict)

if __name__ == "__main__":
    main()