import random
import pprint
from sklearn import datasets

import ROC
import NaiveBayes
import CrossValidator

# Add the target label at the end of the dataset. Needed to shuffle the data easily.
def concateTargetWithDataset(dataset, targetDataset):
  data = list()
  for index, instance in enumerate(dataset):
    tmp = list(instance)
    tmp.append(targetDataset[index])
    data.append(tmp)
  return data

# Just the same but whitout a different flower (3) to stay on a binary classification.
def mainWhitoutFirstFlower():
  irisData = datasets.load_iris()
  irisData.data = irisData.data[:-50]
  irisData.target = irisData.target[:-50]
  newDataset = concateTargetWithDataset(irisData.data, irisData.target)

  naiveBayes = NaiveBayes.NaiveBayes()
  crossValidator = CrossValidator.CrossValidator(algo=naiveBayes, dataset=newDataset, nbFolds=10)
  _scoresByFold, meanAccuracy, rocData = crossValidator.score()
  print('Accuracy: %.2f%%' % meanAccuracy)

  roc = ROC.ROC()
  roc.rocCurve(rocData)
  roc.showROC()

# Just the same but whitout a different flower (2) to stay on a binary classification.
def mainWhitoutMiddleFlower():
  irisData = datasets.load_iris()
  irisData.data = [instance for index, instance in enumerate(irisData.data) if index < 51 or index > 100]
  irisData.target = list(filter(lambda label: label != 1, irisData.target))
  newDataset = concateTargetWithDataset(irisData.data, irisData.target)

  naiveBayes = NaiveBayes.NaiveBayes()
  crossValidator = CrossValidator.CrossValidator(algo=naiveBayes, dataset=newDataset, nbFolds=10)
  _scoresByFold, meanAccuracy, rocData = crossValidator.score()
  print('Accuracy: %.2f%%' % meanAccuracy)

  roc = ROC.ROC()
  roc.rocCurve(rocData)
  roc.showROC()

# Whitout the last flower (1) to stay on a binary classification.
def mainWhitoutLastFlower():
  irisData = datasets.load_iris()
  irisData.data = irisData.data[50:]
  irisData.target = irisData.target[50:]
  newDataset = concateTargetWithDataset(irisData.data, irisData.target)

  naiveBayes = NaiveBayes.NaiveBayes()
  crossValidator = CrossValidator.CrossValidator(algo=naiveBayes, dataset=newDataset, nbFolds=10)
  _scoresByFold, meanAccuracy, rocData = crossValidator.score()
  print('Accuracy: %.2f%%' % meanAccuracy)

  roc = ROC.ROC()
  roc.rocCurve(rocData)
  roc.showROC()

# Try with all flower, expect with the ROC as the ROC is for binary classification
def mainWIthAllFlower():
  irisData = datasets.load_iris()
  newDataset = concateTargetWithDataset(irisData.data, irisData.target)

  naiveBayes = NaiveBayes.NaiveBayes()
  crossValidator = CrossValidator.CrossValidator(algo=naiveBayes, dataset=newDataset, nbFolds=10)
  _scoresByFold, meanAccuracy, _rocData = crossValidator.score()
  print('Accuracy: %.2f%%' % meanAccuracy)

def mainTestBrainCancer():
  irisData = datasets.load_breast_cancer()
  newDataset = concateTargetWithDataset(irisData.data, irisData.target)

  naiveBayes = NaiveBayes.NaiveBayes()
  crossValidator = CrossValidator.CrossValidator(algo=naiveBayes, dataset=newDataset, nbFolds=10)
  _scoresByFold, meanAccuracy, rocData = crossValidator.score()
  print('Accuracy: %.2f%%' % meanAccuracy)

  roc = ROC.ROC()
  roc.rocCurve(rocData)
  roc.showROC()

if __name__ == "__main__":
    mainWhitoutFirstFlower()
    mainWhitoutMiddleFlower()
    mainWhitoutLastFlower()
    mainWIthAllFlower()
    mainTestBrainCancer()