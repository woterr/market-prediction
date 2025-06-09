import numpy as np

def normalisation(X):
    sd = X.std(axis=0)
    mean = X.mean(axis=0)
    X_new = (X - mean)/(sd + 1e-7)
    return X_new

def sigmoid(z):
    return (1/( 1 + ( np.exp(-z) ) ) )


def cost_fn(fX, Y, w=None, regularisation_rate=None):
    m = Y.shape[0]
    epsilon = 1e-8
    loss = - (1/m) * np.sum(
        Y * np.log(fX + epsilon) + (1 - Y) * np.log(1 - fX + epsilon)
    )
    if regularisation_rate != None:
        reg_term = (regularisation_rate / (2 * m)) * np.sum(np.square(w))
        return loss + reg_term
    else:
        return loss


def train(X, Y, lr=1e-1, epochs=10000, lambda_=1e-2):
    m, n = X.shape
    b = 0
    w = np.zeros(n)


    for epoch in range(epochs):
        z = np.dot(X, w) + b
        fX = sigmoid(z)

        cost = cost_fn(fX, Y, w, lambda_)
        if epoch%100 == 0:
            print(f'Epoch: {epoch}, Cost: {cost}')

        dw = (1/m) * np.dot(X.T, (fX - Y)) + ((lambda_*w)/m)
        db = (1/m) * np.sum(fX - Y)

        w -= lr*dw
        b -= lr*db

    return w, b


def predict(X, w, b, threshold=0.45):
    z = np.dot(X, w) + b
    fX = sigmoid(z)
    prediction = (fX >= threshold).astype(int) 

    return prediction, fX


def predited_loss(fX, Y):
    epsilon = 1e-8
    losses = - (Y * np.log(fX + epsilon) + (1 - Y) * np.log(1 - fX + epsilon))
    for i in range(len(losses)):
        if losses[i] < 0:
            losses[i] = 0
        else:
            losses[i] = 1
    return losses