from __future__ import print_function
import torch
from io import open
import re
from torch.autograd import Variable
from torch import nn
import unicodedata
import numpy as np
import random
from itertools import chain


def variableFromSentence(corpus, sentence):
    indexes = corpus.sentence_to_index(sentence)
    result = Variable(torch.LongTensor(indexes).view(-1,1))
    return result


def variablesFromPair(pair):
    global source_corpus
    global target_corpus
    input_variable = variableFromSentence(source_corpus, pair[0])
    target_variable = variableFromSentence(target_corpus, pair[1])
    return [input_variable, target_variable]

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

class Corpus():
    def __init__(self, input_lines, n_train=5000):
        self.SOS = 0
        self.EOS = 1
        self.idx_word, \
        self.word_idx = self.parse_words(input_lines)
        self.n_train = n_train
        
        self.parse_words(input_lines)
        self.corpus_size = len(self.idx_word)
        self.lines = [l.strip().lower() for l in input_lines]
        self.training = [self.sentence_to_index(l) for l in self.lines]
        
    def parse_words(self, lines):
        sls = lambda s: s.strip().lower().split(" ")
        words = ["<SOS>", "<EOS>"] + sorted(set(list( \
                chain(*[sls(l) for l in lines]))))
        idx_word = dict(list(enumerate(words)))
        word_idx = dict(zip(words, list(range(len(words)))))
        
        return idx_word, word_idx
    
    def sentence_to_index(self, s):
        words = s.split(" ")
        indices = [self.word_idx[word] for word in words if word != ""]
        return indices
    
    def index_to_sentence(self, indices):
        return " ".join(self.idx_word[idx] for idx in indices)

    
# you will use this class later, consider using ipython terminal
# to get used to initializing Parameters, Variables
class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weights = nn.Parameter(torch.rand(self.input_size, self.output_size))
        self.bias = nn.Parameter(torch.rand(self.output_size))
        
        
    def forward(self, input_var, use_relu=False):
        # standard linear layer, just an affine transform with no nonlinearity 
        #  for this lab
        # use torch.matmul not mm
        if use_relu:
            # apply relu
            pass
        else:
            return tf.matmul(input_var, self.weights) + self.bias


"""
encoder topology
-standard GRU, embed each word before submitting it.
-use GRUCell not GRU

Never use one hot encodings! Your input tensors to the GRU should have (1,1,hidden) size
-pytorch typically works with 3d (seq_len, batch_size, hidden_dim) tensors, unlike tensorflow's list of tensors.
"""

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(Encoder, self).__init__()
        print("encoding...")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embed = nn.Embedding(self.input_size, self.hidden_size) # needs parameters
        self.GRUCell = nn.GRUCell(self.input_size, self.hidden_size) # needs parameters
                
    def forward(self, input_variable):
        embedded = self.embed(input_variable).view(1,1,-1)
        hidden = Variable(torch.zeros(1,1,self.hidden_size))
        for i in range(self.n_layers):
            hidden = self.GRUCell(embedded[i], hidden)
        return hidden, hidden

"""
GRU is initialized to the number of layers
-run it one time step at a time using tensors of shape (1,1,hidden_size)
-use zero's as the initial hidden state

Use teacher forcing to initially establish word-word connections
-recommend around .5 to .7

without teacher forcing
next_input = tensor([[SOS]])
next_hidden = hidden
for i in 0..len(input_sequence):
  embed(next_input), h_i-1 -> GRU -> output, h_i
  output[-1] -> LinearLayer (to number of words in English corpus) -> SoftMax -> probabilities
  probabilities -> argmax -> next_input
  if next_input = EOS:
    break

with teacher forcing, helps to form one to one connections between words 
embedded = embed(reference_var)
next_hidden = hidden
for i in 0..len(embedded):
  embedded[i], h_i-1 -> GRU -> output, h_i
  output[-1] -> LinearLayer (to number of words in English corpus) -> SoftMax -> probabilities

Return the probabilities (for the loss) and predictions (for printing and easy later evaluation) whether or not using teacher forcing.
"""

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, target_vocab_size, n_layers=1, max_target_length=30):
        super(Decoder, self).__init__()
        print("decoding...")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_vocab_size = target_vocab_size
        self.n_layers = n_layers
        self.max_target_length = max_target_length
        
        # initialize GRU, a Linear Layer
        self.embed = nn.Embedding(self.target_vocab_size, self.hidden_size)
        self.GRUCell = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.out = Linear(self.hidden_size, self.target_vocab_size)

    
    def forward(self, context, target_variable=None, encoder_outputs=None):
        print(context, target_variable)
        embedded = self.embed(context).view(1,1,-1)
        output = embedded
        hidden = Variable(torch.zeros(1,1,self.hidden_size))
        for i in range(self.n_layers):
            hidden = self.GRUCell(nn.functional.relu(output), hidden)
        return nn.LogSoftmax(self.out(hidden)), hidden



def train(input_variable, target_variable,encoder, decoder, encoder_optim, decoder_optim,loss_fn, max_seq_len=30):
    print("training...")
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    decoder_words = []

    encoder_outputs = Variable(torch.zeros(max_seq_len,encoder.hidden_size))

    loss = 0

    for i in range(input_length):
        print(i, "encoder loop")
        # encoder_output = input_variable[i]
        encoder_output, encoder_hidden = encoder(input_variable[i])
        encoder_outputs[i] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[0]]))

    use_teacher_forcing = np.random.random() < teacher_forcing_ratio

    if use_teacher_forcing:
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, encoder_output, encoder_outputs)
            loss += loss_fn(decoder_output, target_variable[i])
            decoder_input = target_variable[i] # This is the teacher forcing portion
    else:
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]


            decoder_input = Variable(torch.LongTensor([[ni]]))

            loss += loss_fn(decoder_output, target_variable[i])
            decoder_words.append(decoder_output)
            if ni == 1:
                break
    loss.backward()

    encoder_optim.step()
    decoder_optim.step()

    return loss.data[0]/target_length, decoder_words



def get_loss(output_probs, correct_indices):
    """ params:
         output_probs: a list of Variable (Not Tensor)
         correct_indices: a list or tensor of type int with the same length """
    
    # You can batch this part if you want by concatenating the output_probs
    # convert correct_indices to a (seq_len, 1) Tensor if batching or list of tensors
    # Use NLLoss as it takes probabilities
    # sanity check: loss should be a Variable

def print_output(teacher_forced, source_indices, predicted_indices, reference_indices):
    global source_corpus, target_corpus
    print("Iteration %d", end=" ")
    if teacher_forced:
        print("using teacher forcing")
    print ("In:    ", source_corpus.index_to_sentence(source_indices))
    print ("Out:   ", target_corpus.index_to_sentence(predicted_indices))
    print ("Ground:", target_corpus.index_to_sentence(predicted_indices))

# can combine encoder and decoder
def train_iters(encoder, decoder, training_pairs, testing_pairs, 
                source_corpus, target_corpus, teacher_forcing_ratio, 
                epoch_size, learning_rate, batch_size, print_every):
    """
    You may want to lower the teacher forcing ratio as the number 
      of epochs progresses as it starts to learn word-word connections.
    
    In PyTorch some optimizers don't allow for decaying learning rates
    -thankfully initializing new optimizers is trivial
    -You may want to use a learning rate schedule instead of decay
    """


    print_loss_total = 0
    # initialize the optimizers
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()
    
    batched_loss = 0
    for i in range(n_epochs):

        # use_teacher_forcing = np.random.random() < teacher_forcing_ratio

        # consider whether or not to use teacher forcing on printing iterations
        # use_teacher_forcing = use_teacher_forcing or (i % print_every == 0)
        pair = variablesFromPair(random.choice(training_pairs))
        input_variable, target_variable = pair[0], pair[1]

        loss, sentence = train(input_variable, target_variable, encoder, decoder, encoder_optim, decoder_optim, loss_fn)

        print_loss_total += loss
        
        # convert source_var and reference_var to Tensors then Variables
        # run source_var through the encoder
        # if use_teacher_forcing input the target variable to decoder
        # compute the loss between the predicted probability distributions and the target indexed tensor

        # implement the batch update as shown in the spec

        # print results
        if i % print_every == 0:
            print_output(use_teacher_forcing, source_idc, sentence, target_idc)
            
if __name__ == "__main__":


    source_file = "data/es.txt"
    target_file = "data/en.txt"
    temp_source_lines = list(open(source_file, encoding='utf-8'))
    temp_target_lines = list(open(target_file, encoding='utf-8'))
    source_lines = []
    target_lines = []
    for line in temp_source_lines:
        source_lines.append(normalizeString(line))

    for line in temp_target_lines:
        target_lines.append(normalizeString(line))


    # very crude
    s="abcdefghijklmnopqrstuvwxyz"

    training_pairs = zip(source_lines, target_lines)

    max_seq_len = 30
    keep_pair_if = lambda pair: len(pair[0].split(" ")) < max_seq_len and \
                             len(pair[1].split(" ")) < max_seq_len

    training_pairs = filter(keep_pair_if, training_pairs)
    source_lines, target_lines = zip(*training_pairs)

    source_corpus = Corpus(source_lines)
    target_corpus = Corpus(target_lines)
    n_spanish = source_corpus.corpus_size
    n_english = target_corpus.corpus_size

    # accumulate gradients batch_size times before updating
    epoch_length = np.minimum(len(training_pairs), 4000)
    batch_size = 20 # rather iterations between optimizer updates
    max_seq_len = 30 # watch the number of sentences drop
    n_layers = 2
    learning_rate = .01
    print_every = batch_size
    n_epochs = 30

    input_size = 30
    hidden_size = 256
    teacher_forcing_ratio = .5
    n_test = 5000


    indexed_pairs = zip(source_corpus.training, target_corpus.training)

    # support for training yet to come
    encoder = Encoder(input_size, hidden_size, n_layers)
    decoder = Decoder(input_size, hidden_size, n_english, n_layers, max_seq_len)

    training_pairs = training_pairs[:-n_test]
    testing_pairs = training_pairs[-n_test:]
    train_iters(encoder, decoder, training_pairs,testing_pairs,source_corpus,target_corpus,teacher_forcing_ratio,epoch_length,learning_rate, batch_size,print_every)
