import matplotlib.pyplot as plt
import numpy as np

def plot_misclassified_images(img_data, classes, img_name):
  figure = plt.figure(figsize=(10, 10))
  
  num_of_images = len(img_data)
  for index in range(1, num_of_images + 1):
      img = img_data[index-1]["img"] / 2 + 0.5     # unnormalize
      plt.subplot(5, 5, index)
      plt.axis('off')
      plt.imshow(np.transpose(img, (1, 2, 0)))
      plt.title("Predicted: %s\nActual: %s" % (classes[img_data[index-1]["pred"]], classes[img_data[index-1]["target"]]))
  
  plt.tight_layout()
  plt.savefig(img_name)

def plot_graph(data, metric):
    fig = plt.figure(figsize=(7, 7))
    
    plt.title(f'Validation %s' % (metric))
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.plot(data)
    plt.show()
    
    fig.savefig(f'val_%s_change.png' % (metric.lower()))
    

def plot_metrics(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")