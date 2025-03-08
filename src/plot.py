import matplotlib.pyplot as plt

def plot_loss(loss_per_epoch, label='Training loss', title='Training loss per epoch'):
    epochs = range(1, len(loss_per_epoch) + 1)
    plt.plot(epochs, loss_per_epoch, 'bo', label=label)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
