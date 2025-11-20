import losses
import modules
import trainer
import optimizers
import mnist_loader
import numpy as np


def one_hot_encoding(y: np.ndarray, num_classes: int = None) -> np.ndarray:
    y = np.array(y, dtype='int').ravel()
    num_classes = num_classes or np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype='float32')
    categorical[np.arange(n), y] = 1
    return categorical


def preprocess_input(images: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # reshape to one column and normalize
    images = images.astype('float32').reshape(-1, 784, 1) / 255
    # one-hot encoding for labels
    labels = one_hot_encoding(labels).reshape(-1, 10, 1)
    return images, labels


x_train, y_train, x_test, y_test = mnist_loader.download_mnist()
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")


x_train, y_train = preprocess_input(x_train, y_train)
x_test, y_test = preprocess_input(x_test, y_test)

model = modules.Sequential([
    modules.Linear(784, 100),
    modules.Tanh(),
    modules.Linear(100, 50),
    modules.Tanh(),
    modules.Linear(50, 10),
    modules.Softmax()
])

'''

trainer.train(
    model,
    x_train[:3000],
    y_train[:3000],
    losses.CrossEntropy(),
    optimizers.GD(model, learning_rate=0.001),
    epochs=30,
)
'''

'''
trainer.train_batch_gd(
    model,
    x_train[:3000],
    y_train[:3000],
    losses.CrossEntropy(),
    optimizers.GD(model, learning_rate=0.001),
    epochs=30,
)
'''


'''
trainer.train_batch(
    model,
    x_train[:3000],
    y_train[:3000],
    losses.CrossEntropy(),
    optimizers.GD(model, learning_rate=0.001),
    epochs=30,
    batch_size=1000,
)
'''




# Compute accuracy over the whole test set
score = 0
for x, y in zip(x_test, y_test):
    true = np.argmax(y)
    pred = np.argmax(model.forward(x))
    if true == pred:
        score += 1

print(f"Score: {100 * score / len(x_test):.2f}%")

