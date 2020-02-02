import pprint
from sklearn import datasets
from sklearn.model_selection import train_test_split

import NaiveBayes

def main():
  irisData = datasets.load_iris()
  irisData.data = irisData.data[:-50]
  irisData.target = irisData.target[:-50]

  X_train, X_test, y_train, y_test = train_test_split(irisData.data, irisData.target, test_size=0.2, random_state=0)

  naiveBayes = NaiveBayes.NaiveBayes(trainData=X_train, trainTarget=y_train, testData=X_test, testTarget=y_test)
  naiveBayes.fit()

if __name__ == "__main__":
    main()