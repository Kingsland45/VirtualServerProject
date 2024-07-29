# Austin Kingsland CSE-432

import numpy as np
from urllib import request
import gzip
import pickle

filename = [
    ["training_images","train-images-idx3-ubyte.gz"],
    ["test_images","t10k-images-idx3-ubyte.gz"],
    ["training_labels","train-labels-idx1-ubyte.gz"],
    ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")

def load():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

def init():
    download_mnist()
    save_mnist()

if __name__ == '__main__':
    init()

# Loading the data
TRimg, TRlab, TSimg, TSlab = load()

# Normalizing the data
TRimg = TRimg.astype('float32') / 255
TSimg = TSimg.astype('float32') / 255

print(len(TRimg), len(TRlab), len(TSimg), len(TSlab))
print(len(TRimg[0]), len(TRlab), len(TSimg[0]), len(TSlab))
print(TRlab[10])

# Initialize the parameters for the three-layer neural network
D = 784  # input size
h1 = 100  # size of first hidden layer
h2 = 50   # size of second hidden layer
K = 10    # output size
step_size = 0.1
reg = 0.001

W1 = 0.01 * np.random.randn(D, h1)
b1 = np.zeros((1, h1))
W2 = 0.01 * np.random.randn(h1, h2)
b2 = np.zeros((1, h2))
W3 = 0.01 * np.random.randn(h2, K)
b3 = np.zeros((1, K))

# Training the network
Epoc = 10
BatchSize = 32

for i in range(Epoc):
    for j in range(0, 60000, BatchSize):
        X = TRimg[j:j+BatchSize]
        Y = TRlab[j:j+BatchSize]
        num_examples = X.shape[0]

        # Forward propagation
        hidden_layer1 = np.maximum(0, np.dot(X, W1) + b1)
        hidden_layer2 = np.maximum(0, np.dot(hidden_layer1, W2) + b2)
        scores = np.dot(hidden_layer2, W3) + b3

        # Loss computation
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(num_examples), Y])
        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
        loss = data_loss + reg_loss

        # Backpropagation
        dscores = probs
        dscores[range(num_examples), Y] -= 1
        dscores /= num_examples

        dW3 = np.dot(hidden_layer2.T, dscores)
        db3 = np.sum(dscores, axis=0, keepdims=True)
        dhidden2 = np.dot(dscores, W3.T)
        dhidden2[hidden_layer2 <= 0] = 0

        dW2 = np.dot(hidden_layer1.T, dhidden2)
        db2 = np.sum(dhidden2, axis=0, keepdims=True)
        dhidden1 = np.dot(dhidden2, W2.T)
        dhidden1[hidden_layer1 <= 0] = 0

        dW1 = np.dot(X.T, dhidden1)
        db1 = np.sum(dhidden1, axis=0, keepdims=True)

        dW3 += reg * W3
        dW2 += reg * W2
        dW1 += reg * W1

        W1 -= step_size * dW1
        b1 -= step_size * db1
        W2 -= step_size * dW2
        b2 -= step_size * db2
        W3 -= step_size * dW3
        b3 -= step_size * db3

# Testing the network
hidden_layer1 = np.maximum(0, np.dot(TSimg, W1) + b1)
hidden_layer2 = np.maximum(0, np.dot(hidden_layer1, W2) + b2)
scores = np.dot(hidden_layer2, W3) + b3
predicted_class = np.argmax(scores, axis=1)
print("training accuracy: %.2f" % (np.mean(predicted_class == TSlab)))
