import pkuseg
import torch
from string import punctuation
from format_utils import format_text, get_train_on_gpu
import sys


# method to predict the sentiment of chinese text


def predict(text, no_stopwords):
    print(f'Text: {text}')

    feature_tensor = format_text(text, no_stopwords)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    # move to gpu if on gpu
    if(get_train_on_gpu()):
        feature_tensor = feature_tensor.cuda()

    # get output from model
    output, h = net(feature_tensor, h)

    # normalize output probs to 0 or 1
    pred = torch.round(output.squeeze())

    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    if pred == 0:
        print('negative')
        return 'Text is negative'
    elif pred == 1:
        print('positive')
        return 'Text is positive'


###########################################

batch_size = 50


net = torch.load('models/torch_rnn_trained.model',
                 map_location=torch.device('cpu'))

predict(sys.argv[1], sys.argv[2])
