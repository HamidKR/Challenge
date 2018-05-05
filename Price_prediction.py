from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt


start = dt.datetime(1995, 1, 1)
end = dt.date.today()
data = web.DataReader('CAT', 'quandl', start, end)


data.drop(data.columns[[ 3, 4, 5, 6, 7, 8, 9, 11]], axis=1, inplace=True)




main_data = pd.DataFrame()
main_data = main_data.join(data, how='outer')
print(main_data.head())


main_data['Open'] = main_data['Open'] / 1000
main_data['High'] = main_data['High'] / 1000
main_data['Low'] = main_data['Low'] / 1000
main_data['AdjClose'] = main_data['AdjClose'] / 1000
main_data.head()



def loading_data(stock_data, sLen):
    seqLen = sLen + 1
    frame = []

    data = stock_data.as_matrix()
    for i in range(len(data) - seqLen):
        frame.append(data[i: i + seqLen])

    frame = np.array(frame)

    split = round(0.95 * frame.shape[0])

    train = frame[:int(split), :]
    X_train = train[:, :-1]
    Y_train = train[:, -1][:, -1]
    X_test = frame[int(split):, :-1]
    Y_test = frame[int(split):, -1][:, -1]


    return [X_train, Y_train, X_test, Y_test]



def make_model(layers):
    model_seq = Sequential()

    model_seq.add(LSTM(120, input_shape=(layers[1], layers[0]), return_sequences=True))

    model_seq.add(Dropout(.2))

    model_seq.add(LSTM(60, input_shape=(layers[1], layers[0]), return_sequences=False))

    model_seq.add(Dropout(.2))

    model_seq.add(Dense(15, kernel_initializer='uniform', activation='relu'))
    model_seq.add(Dense(layers[2], kernel_initializer='uniform', activation='relu'))

    model_seq.compile(loss='mse', optimizer='adam', metrics=['mse', 'mape'])
    return model_seq

#prediction length
plen = 5

X_train, Y_train, X_test, Y_test = loading_data(main_data[::-1], plen)


model = make_model([4, plen, 1])


model.fit( X_train, Y_train, batch_size=500, epochs=500, validation_split=0.1, verbose=1)

train_eval = model.evaluate(X_train, Y_train, verbose=1)
print(train_eval)
print('Train MSE: %f MSE' % (train_eval[0],))

test_eval = model.evaluate(X_test, Y_test, verbose=1)
print(test_eval)
print('Test MSE: %f MSE' % (test_eval[0]))


error = []
prediction = model.predict(X_test)
for i in range(len(prediction)):
    prediction[i][0] = prediction[i][0]* 1000
y_test_actual = Y_test * 1000

for i in range(len(Y_test)):
    error.append((y_test_actual[i] - prediction[i][0]) ** 2)
totalErr = sum(error)
mse = totalErr / len(Y_test)
print(mse)



plt.plot(prediction, color='red', label='prediction')
plt.plot(y_test_actual, color='blue', label='y_test')
plt.legend(loc='lower right')
plt.show()