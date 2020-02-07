import matplotlib.pyplot as plt
import pprint

class ROC:
  def __init__(self):
    self.tpr = list()
    self.fpr = list()

  # Calculate the ROC with the information from the prediction given by the cross validator.
  def rocCurve(self, rocData):
    tp, tn, fp, fn = 0, 0, 0, 0

    for index in range(len(rocData)):
      # Get the negative class.
      class0 = min(rocData[index][1])
      # Get the positive class.
      class1 = max(rocData[index][1])
      # Get the predicted class by the Naive Bayes.
      predictClass = rocData[index][0]
      # Get the target label expected.
      target = rocData[index][2]

      # Main if forest to fill the appropriate variable for the TPR and the FPR.
      if (predictClass == class1 and target == predictClass):
        tp += 1
      elif (predictClass == class1 and target is not predictClass):
        fp += 1
      elif (predictClass == class0 and target == predictClass):
        tn += 1
      elif (predictClass == class0 and target is not predictClass):
        fn += 1
      # Based on wiki, calculate the TRP with the previous variable.
      self.tpr.append(tp / (tp + fn))
      # Same as TPR but for FPR.
      self.fpr.append(fp / (fn + tp))

  # Display the ROC Curve with matplot.    
  def showROC(self):
    lw = 1.5
    plt.plot(self.fpr, self.tpr, color='darkorange', lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()