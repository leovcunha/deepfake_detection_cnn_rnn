import cv2
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.metrics import confusion_matrix

#Function to plot the image
def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype(int))
    plt.show()
    
def calculate_accuracy(outputs, targets):
    
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1,-1))
    accuracy = (correct.float().sum().item() / batch_size) * 100
    return accuracy

class AverageMeter(object):
    '''Computes and stores average and current value
    '''
    def __init__(self):
        self.reset()
    def reset(self):
        self.count = 0
        self.sum = 0
        self.val = 0
        self.avg = 0

    def update(self, val, n = 1): 
        self.count += n
        self.sum += val * n
        self.val = val
        self.avg = self.sum / self.count
        
def print_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.ylabel('Actual label', size = 20)
    plt.xlabel('Predicted label', size = 20)
    plt.xticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.yticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.ylim([2, 0])
    plt.show()
    calculated_acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+ cm[1][1])
    print('True positives = ', cm[0][0])
    print('False positives = ', cm[0][1])
    print('False negatives = ', cm[1][0])
    print('True negatives = ', cm[1][1])
    print('\n')
    print('Accuracy: ',calculated_acc*100)
    print('Precision: ', cm[0][0]/(cm[0][1] + cm[0][0]))
    print('Recall: ', cm[0][0]/(cm[1][0] + cm[0][0]))
    
    
def plot_loss(train_loss_avg, test_loss_avg, num_epochs):
    '''
    Plots loss based on parameters
    
    Parameters:
    train_loss_avg: train loss recorded during model training
    test_loss_avg: validation loss recorded during model training 
    '''
    loss_train = train_loss_avg
    loss_val = test_loss_avg

    epochs = range(num_epochs)
    plt.plot(epochs, loss_train, 'r', label='Training')
    plt.plot(epochs, loss_val, 'g', label='validation')
    plt.title('Training and Validation losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_accuracy(train_accuracy, test_accuracy, num_epochs):

    '''
    Plots loss based on parameters
    
    Parameters:
    train_accuracy: train loss recorded during model training
    test_accuracy: validation loss recorded during model training 
    '''

    acc_train = train_accuracy
    acc_val = test_accuracy
    epochs = range(num_epochs)
    plt.plot(epochs, acc_train, 'r', label='Training ')
    plt.plot(epochs, acc_val, 'g', label='validation ')
    plt.title('Training and Validation accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()