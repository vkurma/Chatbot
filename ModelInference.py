from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

tf.enable_eager_execution()

import unicodedata
import re
import numpy as np
import os
import io
import time
import pickle

from collections import Counter
from itertools import chain
from itertools import dropwhile


MAX_SEQUENCE_LENGTH=25
path_to_file1 = 'cleanQuestions'
path_to_file2 = 'cleanAnswers'
WORDS_FREQ_THRESHOLD = 8

BATCH_SIZE = 128#64
embedding_dim = 256
units = 512 #1024
MAX_GRADIENT_VAL = 10.0
max_length_targ = 25
max_length_inp = 25

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

     #Remove URLS
    w = re.sub(r"http\S+", "", w)
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(paths, num_examples):
    questions = []
    answers = []
    lines = []
    with open (paths[0], 'rb') as fp:
        lines = pickle.load(fp)
    for line in lines:
        questions.append(preprocess_sentence(line))
    with open (paths[1], 'rb') as fp:
        lines = pickle.load(fp)
    for line in lines:
        answers.append(preprocess_sentence(line))
    return (questions, answers)

def removeWordsWithFreqLessThanK(wordCounter, k):
    cnt = 0
    for word in wordCounter:
        if wordCounter[word] < k:
            cnt += 1
            # del wordCounter[word]
            # print(word, wordCounter[word])

    print("removing ", cnt, " out of ", len(wordCounter), len(wordCounter) - cnt)

    for key, count in dropwhile(lambda key_count: key_count[1] >= k, wordCounter.most_common()):
        del wordCounter[key]

    print("final count: ", len(wordCounter))


class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        print("no of lines passed to class", len(self.lang))
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.wordCounter = {}

        self.create_index()

    def create_index(self):
        #     for phrase in self.lang:
        #       self.vocab.update(phrase.split(' '))

        #     self.vocab = sorted(self.vocab)

        self.wordCounter = Counter(chain.from_iterable(map(lambda x: x.split(' '), self.lang)))
        print(len(self.wordCounter))
        removeWordsWithFreqLessThanK(self.wordCounter, WORDS_FREQ_THRESHOLD)

        self.vocab = sorted(set(list(self.wordCounter.keys())))
        self.word2idx['<pad>'] = 0
        self.word2idx['<unk>'] = 1
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 2

        for word, index in self.word2idx.items():
            self.idx2word[index] = word

def BiLSTM(units):
    #if tf.test.is_gpu_available():
	lstm = tf.keras.layers.LSTM(units,
									 return_sequences=True,
									 return_state=True,
									 recurrent_initializer='glorot_uniform')

	return tf.keras.layers.Bidirectional(lstm)
    #else:
        #return "No GPU!!!"
        # return tf.keras.layers.GRU(units,
        #                           return_sequences=True,
        ##                           return_state=True,
        #                           recurrent_activation='sigmoid',
        #                          recurrent_initializer='glorot_uniform')


def LSTM(units):
    # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
    # the code automatically does that.
    #if tf.test.is_gpu_available():
	lstm = tf.keras.layers.LSTM(units,
									 return_sequences=True,
									 return_state=True,
									 recurrent_initializer='glorot_uniform')

	return lstm
    #else:
        #return "No GPU!!!"
        # return tf.keras.layers.GRU(units,
        #                           return_sequences=True,
        ##                           return_state=True,
        #                           recurrent_activation='sigmoid',
        #                          recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.bilstm = BiLSTM(self.enc_units)

    def call(self, x, hidden):
        # X shape is [batchsize, maxSeqLen]
        x = self.embedding(x)
        # X shape is [batchsize, maxSeqLen, embedding dimension]
        output, fw_H, fw_C, bw_H, bw_C = self.bilstm(x)
        # output, fw_H, fw_C, bw_H, bw_C = self.bilstm(x, initial_state = hidden)

        # Output shape is [batchsize, maxSeqLen, 2*LSTM Units]
        # fw or bw hidden state shape is [batchsize, LSTM Units]

        ## We concatinate these states because in decoder we can only have a unidirectinal layer. This is because in
        ## decoder we dont the future words.
        final_H = tf.concat((fw_H, bw_H), 1)
        # After concatenation hidden state shape is [batchsize, 2*LSTM Units]
        final_C = tf.concat((fw_C, bw_C), 1)
        return output, final_H, final_C

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units * 2
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(self.dec_units)
        ## We use this layer to pass the LSTM output to get the logits for each word, hence the dimension - vocab_size.
        ## These logits are for each word being the next word in output sentence.
        ## We pass this fc out to error function which takes logits as input.
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.W3 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, dec_H, dec_C, enc_output):
        # enc_output shape == (batch_size, max_length, 2*encoder units)

        # hidden shape == (batch_size, decoder units)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size or decoder units)
        # we are doing this to perform addition to calculate the score
        dec_H_time_axis = tf.expand_dims(dec_H, 1)
        dec_C_time_axis = tf.expand_dims(dec_C, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H) +FC(C)) to self.V
        # The out put of a lstm layer is not a softmax(not logits, so pass them through dense layers)
        # and get the Logits and then pass to a dense layer.
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(dec_H_time_axis) + self.W3(dec_C_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, H, C = self.lstm(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)

        return x, H, C, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))

def getLangObj(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        return obj

pairs = create_dataset([path_to_file1,path_to_file2], None)

inp_lang = LanguageIndex(pairs[0])
targ_lang = LanguageIndex(pairs[1])

vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.compat.v1.train.AdamOptimizer()

def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

checkpoint_dir = './training_checkpoints2'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

print("########################################################################")
print("restored training checkpoint")
print("########################################################################")

def evaluate(sentence):

    sentence = preprocess_sentence(sentence)

    # inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
    inputs = []
    for i in sentence.split(' '):
        if (i in inp_lang.word2idx):
            inputs.append(inp_lang.word2idx[i])
        else:
            inputs.append(inp_lang.word2idx['<unk>'])

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_output, enc_hidden_H, enc_hidden_C = encoder(inputs, hidden)
    dec_hidden_H = enc_hidden_H
    dec_hidden_C = enc_hidden_C

    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden_H, dec_hidden_C, attention_weights = decoder(dec_input, dec_hidden_H, dec_hidden_C,
                                                                             enc_output)

        # storing the attention weigths to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.idx2word[predicted_id] + ' '

        if(targ_lang.idx2word[predicted_id] == '<end>'):
            result = result.replace('<end>',' ')
            result = result.replace('<unk>', ' ')
            if(result.isspace()):
                result = "sorry I didnt get it"
            print("prediction is :", result)
            return result

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
	
    result = result.replace('<end>', ' ')
    result = result.replace('<unk>', ' ')
    if(result.isspace()):
        result = "sorry I didnt get it"
    print("prediction is :", result)
    return result
    #return result.replace('<end>', '')

# sentence = "I am looking for red cars"
#
# op, encOp = evaluate(sentence)
#
# encOp = encOp


