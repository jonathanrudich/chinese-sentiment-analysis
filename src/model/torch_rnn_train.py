import torch
import torch.nn as nn
import numpy as np

print('loading model and train data...')
# load model from file
net = torch.load('models/torch_rnn.model')

# load data from file
train_loader = torch.load('data/processed/torch_rnn_train.loader')
valid_loader = torch.load('data/processed/torch_rnn_validate.loader')

# loss and optimization functions
lr = 0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# training params
epochs = 8

counter = 0
print_every = 50
clip = 5  # gradient clipping
batch_size = 50

# check for gpu
print('checking if gpu avaliable...')
train_on_gpu = torch.cuda.is_available()

print('training on gpu') if train_on_gpu else print(
    'no gpu avaliable, training on cpu')

# move model to gpu, if avaliable
if train_on_gpu:
    net.cuda()

print('training model...')
net.train()

# train for some number of epochs
for e in range(epochs):

    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # create new variables for hidden state to prevent
        # backpropping through entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()

        # clip_grad_norm helps prevent the exploding gradient problem in RNNs
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:

            # get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # create new variables for hidden state
                val_h = tuple([each.data for each in h])

                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))


# save trained model
print('saving trained model...')
torch.save(net, 'models/torch_rnn_trained.model')
