import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, drop_p=0.2):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # Embedded layer
        self.embed = nn.Embedding(vocab_size, embed_size)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=drop_p)

        # Fully Connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x hidden_size,
        # initialized to zero, for hidden state and cell state of LSTM
        # Why do we need batch_size here?
        # Because we are using batch_first=True in the LSTM layer
        # Does that mean there are batch_size number of hidden states for each unit?
        # Yes, that's right
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))

    def forward(self, features, captions):
        # features.shape: (batch_size, embed_size)
        # captions.shape: (batch_size, captions_length)

        # Initialize the hidden state
        # Why do we do this?
        # Because we are using batch_first=True in the LSTM layer
        # Does that mean initialize the hidden state for each data in the batch?
        # Yes, that's right
        self.hidden_state = self.init_hidden(features.shape[0])

        # Remove the last word from the captions
        # Why do we do this?
        # Because we don't want to predict the last word
        captions = captions[:, :-1]
        # captions.shape: (batch_size, captions_length - 1)

        # Embed the captions
        embeddings = self.embed(captions)
        # embeddings.shape: (batch_size, captions_length - 1, embed_size)

        # Concatenate the features and the captions
        # matches the input shape of the LSTM layer
        features = features.unsqueeze(1)
        # features.shape: (batch_size, 1, embed_size)
        # Why do we do this?
        # Because we want to concatenate the features and the captions
        # Why do we need captions here?
        # Because we want to predict the next word
        # The tokens in the captions here act like labels for each step?
        # Yes, that's right
        inputs = torch.cat((features, embeddings), 1)
        # inputs.shape: (batch_size, captions_length, embed_size)

        # Pass the inputs through the LSTM
        lstm_out, self.hidden_state = self.lstm(inputs, self.hidden_state)
        # lstm_out.shape: (batch_size, captions_length, hidden_size)

        # Pass the LSTM output through the FC layer
        outputs = self.fc(lstm_out)
        # outputs.shape: (batch_size, captions_length, vocab_size)

        # Output should be designed such that outputs[i,j,k] contains the model's predicted score,
        # indicating how likely the j-th token in the i-th caption in the batch
        # is the k-th token in the vocabulary.
        # softmax
        # Do we need softmax layer?
        # No, we don't need it
        # Why?
        # Because we are using CrossEntropyLoss
        # CrossEntropyLoss combines LogSoftmax and NLLLoss in one single class
        # outputs = F.softmax(outputs, dim=2)
        # outputs.shape: (batch_size, captions_length, vocab_size)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        '''
        accepts pre-processed image tensor (inputs) and returns predicted sentence
        (list of tensor ids of length max_len)
        '''
        # inputs.shape: (batch_size, 1, embed_size)

        # Initialize the hidden state
        self.hidden_state = self.init_hidden(inputs.shape[0])

        # Initialize the predicted sentence
        predicted_sentence = []

        # Loop through the max_len
        for i in range(max_len):
            # Pass the inputs through the LSTM
            lstm_out, self.hidden_state = self.lstm(inputs, self.hidden_state)

            # Pass the LSTM output through the FC layer
            outputs = self.fc(lstm_out)
            # print("outputs.shape:", outputs.shape)
            # outputs.shape: torch.Size([1, 1, 8855]) (batch_size, 1, vocab_size)

            # Get the predicted word
            predicted_word = outputs.argmax(dim=2).item()
            # print("predicted_word:", predicted_word)
            # predicted_word: 0

            # end the loop if the predicted word is <end>
            if predicted_word == 1:
                break

            # Append the predicted word to the predicted sentence
            predicted_sentence.append(predicted_word)

            predicted_word = torch.LongTensor([predicted_word]).unsqueeze(1).to(self.device)
            # print("predicted_word.shape:", predicted_word.shape)
            # predicted_word.shape: torch.Size([1, 1]) (batch_size, 1)

            # Update the inputs
            inputs = self.embed(predicted_word)
            # print("inputs.shape:", inputs.shape)
            # inputs.shape: torch.Size([1, 1, 256]) (batch_size, 1, embed_size)

        return predicted_sentence
