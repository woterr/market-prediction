import numpy as np
from data import load_data
from utils import normalisation, train, predict, predited_loss
from graphing_logistic import plot_predictions

treshold = 0.53
lr = 1e-2
epochs = 1000

X_raw, Y= load_data(start="2024-01-01", end="2025-06-06")
X = normalisation(X_raw)
W, B = train(X, Y, lr=lr, epochs=epochs)


X_test_raw, Y_test = load_data(start="2023-8-04", end="2023-12-20")
X_test = normalisation(X_test_raw)
predictions, fX = predict(X_test, W, B, threshold=treshold)
losses = predited_loss(predictions, Y_test)

pred_labels = (fX >= 0.45).astype(int)
unique, counts = np.unique(pred_labels, return_counts=True)
print(dict(zip(unique, counts)))

for i in range(len(predictions)):
    print(f"Day {i}: Predicted : {'UP' if predictions[i] else 'DOWN'}, Confidence = {fX[i]*100:.2f}%\t|\tReal data : {'UP' if Y_test[i] else 'DOWN'} Loss: {losses[i]}")
    
plot_predictions(Y_test, fX, threshold=treshold)