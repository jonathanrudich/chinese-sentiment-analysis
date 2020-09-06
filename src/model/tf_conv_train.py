import pickle
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model

# load data dict
print('loading data dict...')
with open('data/processed/data_dict.pkl', 'rb') as handle:
    data_dict = pickle.load(handle)

# load model
print('loading model...')
model = load_model('models/embedded_conv_1d')

# compile
print('compiling model...')
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# fit model
print('fitting model...')
model.fit(data_dict['x_train'], data_dict['y_train'], epochs=2  0, verbose=2)

# evaluate model
print('evaluating model...')
loss, acc = model.evaluate(data_dict['x_test'], data_dict['y_test'], verbose=2)
print(f'Test Accuracy: {acc * 100}')

# save model
print('saving model...')
model.save('models/embedded_conv_1d_trained')
