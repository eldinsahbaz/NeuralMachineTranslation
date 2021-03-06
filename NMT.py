import pandas as pd, numpy as np, tensorflow as tf, re, time, sys, contractions, _pickle as pickle, os, nltk, random, string, warnings, os, sys
from numpy import newaxis
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.python.layers.core import Dense
from nltk.corpus import stopwords
from multiprocessing import Pool
from collections import Counter
from pprint import pprint
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import *
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from copy import deepcopy

#eager.enable_eager_execution()
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train_original = "train.en"
train_translated = "train.vi"
test_original = "tst2013.en"
test_translated = "tst2013.vi"
word_number_mapping_file = "word_mappings.txt"
processed_original = "translated_numeric.txt"
processed_translated  = "original_numeric.txt"
modelDir = './model/'
modelFileName = 'Eldin_Sahbaz_Model.ckpt'

#this commented filtering code splits the code at the character level rather than the word level
'''
def filter_symbols(original_input, translated_input):
    try:
        zero = lambda text: contractions.fix(text.lower())
        one = lambda text: re.sub('\.\.\.', ' ', zero(text))
        two = lambda text: list(one(text)) #[character for character in list(one(text)) if (character not in ['-', '\\', '/', '.', '—', '…', '...', '?', ',', '<', '>', '\"', ';', ':', '[', ']', '{', '}', '|', '=', '+', '_', '*', '&', '^', '%', '$', '#', '@', '!', '`', '~'])]

        return (two(original_input), two(translated_input))
    except:
        return None

def filter_symbols_test(input_text):
    try:
        zero = lambda text: contractions.fix(text.lower())
        one = lambda text: re.sub('\.\.\.', ' ', zero(text))
        two = lambda text: list(one(text)) #[character for character in list(one(text)) if (character not in ['-', '\\', '/', '.', '—', '…', '...', '?', ',', '<', '>', '\"', ';', ':', '[', ']', '{', '}', '|', '=', '+', '_', '*', '&', '^', '%', '$', '#', '@', '!', '`', '~'])]

        return two(input_text)
    except:
        return None

'''

#In this function, I convert all contractions (converting can't to cannot and so on), tokenize the sentences at the word
#level, lemmatize the words and remove the stop words from the input text but not from the translated text
def filter_symbols(original_input, translated_input):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    table = str.maketrans({key: None for key in string.punctuation})

    try:
        zero = lambda text: contractions.fix(text.lower())
        one = lambda text: re.sub('\.\.\.', '.', zero(text))
        two = lambda text: nltk.word_tokenize(one(text))
        three_1 = lambda text: list(filter(lambda x: x, [lemmatizer.lemmatize(word1.translate(table)) for word1 in two(text) if ((word1 not in stop_words))]))
        three_2 = lambda text: list(filter(lambda x: x, [lemmatizer.lemmatize(word1.translate(table)) for word1 in two(text)]))

        return (three_1(original_input), three_2(translated_input))
    except:
        return None

#This is the same as the function above except for it only takes the input text that is used for testing
def filter_symbols_test(input_text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    table = str.maketrans({key: None for key in string.punctuation})

    try:
        zero = lambda text: contractions.fix(text.lower())
        one = lambda text: re.sub('\.\.\.', '.', zero(text))
        two = lambda text: nltk.word_tokenize(one(text))
        three = lambda text: list(filter(lambda x: x, [lemmatizer.lemmatize(word1.translate(table)) for word1 in two(text) if ((word1 not in stop_words))]))

        return(three(input_text))
    except:
        return None

#Here load the data files, use multiprocessing to apply filtering to all the data, then store the filtered files
def clean_data(original, translated):
    cleaned = None
    with open(original, 'r', encoding="utf8") as file:
        original_data = file.read().split('\n')

    with open(translated, 'r', encoding="utf8") as file:
        translated_data = file.read().split('\n')

    data = list(zip(original_data, translated_data))

    pool = Pool()

    cleaned = pool.starmap(filter_symbols, data)

    pool.close()
    pool.join()

    original_text, translated_text = list(zip(*cleaned))
    original_text = list(filter(lambda y: y, original_text))
    translated_text = list(filter(lambda y: y, translated_text))
    with open("filtered_data", 'wb') as file: pickle.dump(cleaned, file)
    return (original_text, translated_text)

def convert_text(original_text, translated_text, cutoff):
    original_DNS = {'forward':{'<PAD>':0, '<UNK>':1, '<EOS>':2, '<GO>':3}, 'backward':{0:'<PAD>', 1:'<UNK>', 2:'<EOS>', 3:'<GO>'}}
    translated_DNS = {'forward': {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2, '<GO>': 3}, 'backward': {0: '<PAD>', 1: '<UNK>', 2: '<EOS>', 3: '<GO>'}}
    original_words = list()
    translated_words = list()
    stop_words = set(stopwords.words('english'))
    converted_original, converted_translated = list(), list()
    
    #aggregate all the words into a list
    for sentence in original_text: original_words.extend(sentence)
    for sentence in translated_text: translated_words.extend(sentence)

    original_word_frequencies = [x for x in sorted(Counter(original_words).items(), key=lambda x: x[1], reverse=True) if ((x[1] >= cutoff) and (x[0] not in stop_words))]
    translated_word_frequencies = [x for x in sorted(Counter(translated_words).items(), key=lambda x: x[1], reverse=True) if (x[1] >= cutoff)]

    # create mapping for word -> int and for int -> word for the first language
    if original_word_frequencies:
        words, freqs = list(zip(*original_word_frequencies))
        original_DNS['forward'].update(dict(zip(words, list(range(len(original_DNS['forward']), len(words)+len(original_DNS['forward']))))))
        original_DNS['backward'].update({v: k for k, v in original_DNS['forward'].items()})

    # create mapping for word -> int and for int -> word for the second language
    if translated_word_frequencies:
        words, freqs = list(zip(*translated_word_frequencies))
        translated_DNS['forward'].update(dict(zip(words, list(range(len(translated_DNS['forward']), len(words)+len(translated_DNS['forward']))))))
        translated_DNS['backward'].update({v: k for k, v in translated_DNS['forward'].items()})

    #Compute the translation to int for the full text
    for sentence in original_text:
        temp_sentence = list()
        temp_sentence.append(original_DNS['forward']['<GO>'])
        for word in sentence:
            try: temp_sentence.append(original_DNS['forward'][word])
            except : temp_sentence.append(original_DNS['forward']['<UNK>'])
        temp_sentence.append(original_DNS['forward']['<EOS>'])
        converted_original.append(temp_sentence)

    for sentence in translated_text:
        temp_sentence = list()
        temp_sentence.append(translated_DNS['forward']['<GO>'])
        for word in sentence:
            try: temp_sentence.append(translated_DNS['forward'][word])
            except : temp_sentence.append(translated_DNS['forward']['<UNK>'])
        temp_sentence.append(translated_DNS['forward']['<EOS>'])
        converted_translated.append(temp_sentence)

    #These lines of code get some statistics about the dataset
    original_text_lengths, translated_text_lengths, original_unk_counts, translated_unk_counts = list(), list(), list(), list()

    #90th percentile of original text lengths
    for sentence in converted_original: original_text_lengths.append(len(sentence))
    original_text_pd = pd.DataFrame(original_text_lengths, columns=['counts'])
    max_original_length = int(np.percentile(original_text_pd.counts, 90))

    #90th percentile of translated text lengths
    for sentence in converted_translated: translated_text_lengths.append(len(sentence))
    translated_text_pd = pd.DataFrame(translated_text_lengths, columns=['counts'])
    max_translated_length = int(np.percentile(translated_text_pd.counts, 90))

    #5th percentile for minimum text length
    data_pd = pd.DataFrame(original_text_lengths + translated_text_lengths, columns=['counts'])
    min_length = int(np.percentile(data_pd.counts, 5))

    #5th percentile for minimum unknown token limit in original text
    for sentence in converted_original: original_unk_counts.append(Counter(sentence)[original_DNS['forward']['<UNK>']])
    original_pd = pd.DataFrame(original_unk_counts, columns=['counts'])
    unk_original_limit = int(np.percentile(original_pd.counts, 5))

    #5th percentile for minimum unknown token limit in translated text
    for sentence in converted_translated: translated_unk_counts.append(Counter(sentence)[translated_DNS['forward']['<UNK>']])
    translated_pd = pd.DataFrame(translated_unk_counts, columns=['counts'])
    unk_translated_limit = int(np.percentile(translated_pd.counts, 5))

    #truncate all the text and pad them with 0s
    truncated_original_text, truncated_translated_text = list(), list()

    #padding here is done in the front because the string is reversed
    for sentence in converted_original:
        temp = sentence[:max_original_length]
        temp[-1] = original_DNS['forward']['<EOS>']
        temp = list(reversed(temp))
        if len(temp) < max_original_length: temp[0:0] = [original_DNS['forward']['<PAD>']]*(max_original_length-len(temp))
        truncated_original_text.append(temp)

    #padding here is done at the end
    for sentence in converted_translated:
        temp = sentence[:max_translated_length]
        temp[-1] = translated_DNS['forward']['<EOS>']
        if len(temp) < max_translated_length: temp[len(temp):len(temp)] = [translated_DNS['forward']['<PAD>']]*(max_translated_length-len(temp))
        truncated_translated_text.append(temp)

    #remove samples that have too many unknown tokens
    cleaned_truncated_original, cleaned_truncated_translated = list(), list()
    for original, translated in list(zip(truncated_original_text, truncated_translated_text)):
        original_count, translated_count = Counter(original), Counter(translated)

        if ((original_count[original_DNS['forward']['<UNK>']] <= unk_original_limit) and (translated_count[translated_DNS['forward']['<UNK>']] <= unk_translated_limit) and (len(original) >= min_length) and (len(translated) >= min_length)):
            cleaned_truncated_original.append(original)
            cleaned_truncated_translated.append(translated)

    return (original_DNS, translated_DNS, np.array(cleaned_truncated_original), np.array(cleaned_truncated_translated), max_original_length, max_translated_length, min_length, unk_original_limit, unk_translated_limit)

def convert_text_test(original_text, translated_text, original_DNS, translated_DNS, max_original_length, max_translated_length):
    converted_original, converted_translated = list(), list()

    # Compute the translation to int for the full text
    for sentence in original_text:
        temp_sentence = list()
        temp_sentence.append(original_DNS['forward']['<GO>'])
        for word in sentence:
            try: temp_sentence.append(original_DNS['forward'][word])
            except: temp_sentence.append(original_DNS['forward']['<UNK>'])
        temp_sentence.append(original_DNS['forward']['<EOS>'])
        converted_original.append(temp_sentence)

    for sentence in translated_text:
        temp_sentence = list()
        temp_sentence.append(translated_DNS['forward']['<GO>'])
        for word in sentence:
            try: temp_sentence.append(translated_DNS['forward'][word])
            except: temp_sentence.append(translated_DNS['forward']['<UNK>'])
        temp_sentence.append(translated_DNS['forward']['<EOS>'])
        converted_translated.append(temp_sentence)

    # Compute the truncated version of the texts above
    truncated_original_text, truncated_translated_text = list(), list()

    for sentence in converted_original:
        temp = sentence[:max_original_length]
        temp[-1] = original_DNS['forward']['<EOS>']
        temp = list(reversed(temp))
        if len(temp) < max_original_length: temp[0:0] = [original_DNS['forward']['<PAD>']] * (max_original_length - len(temp))
        truncated_original_text.append(temp)

    for sentence in converted_translated:
        temp = sentence[:max_translated_length]
        temp[-1] = translated_DNS['forward']['<EOS>']
        if len(temp) < max_translated_length: temp[len(temp):len(temp)] = [translated_DNS['forward']['<PAD>']] * (max_translated_length - len(temp))
        truncated_translated_text.append(temp)

    return (np.array(truncated_original_text), np.array(truncated_translated_text))

def build_model(num_encoder_tokens, num_decoder_tokens, original_vocab_length, translated_vocab_length, embed_size, nodes, batch_size):
    num_encoder_layers = 1
    num_decoder_layers = 2

    #inputs place holders
    inputs = tf.placeholder(tf.int32, (None, None), 'inputs')  # num_encoder_tokens
    outputs = tf.placeholder(tf.int32, (None, None), 'output')
    targets = tf.placeholder(tf.int32, (None, None), 'targets')
    keep_rate = tf.placeholder(tf.float32, (1), 'keep_rate')
    decoder_size = tf.placeholder(tf.int32, (1), 'decoder_size')

    #embedding variables for the encoder and decoder inputs
    input_embedding = tf.Variable(tf.random_uniform((original_vocab_length, embed_size), -1.0, 1.0), name='enc_embedding')
    output_embedding = tf.Variable(tf.random_uniform((translated_vocab_length, embed_size), -1.0, 1.0), name='dec_embedding')
    encoder_embedding = tf.nn.embedding_lookup(input_embedding, inputs)
    decoder_embedding = tf.nn.embedding_lookup(output_embedding, outputs)

    #bidirectional encoder LSTM with dropout
    prev_input = encoder_embedding
    last_state = None
    for i in range(num_encoder_layers):
        enc_fw_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(nodes), output_keep_prob=keep_rate[0], state_keep_prob=keep_rate[0])
        enc_bw_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(nodes), output_keep_prob=keep_rate[0], state_keep_prob=keep_rate[0])
        ((forward_output, backward_output), (forward_state, backward_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=enc_fw_cell, cell_bw=enc_bw_cell, inputs=prev_input, dtype=tf.float32, scope="encoder_rnn{0}".format(str(i)))
        last_state = tf.contrib.rnn.LSTMStateTuple(c=tf.concat((forward_state.c, backward_state.c), 1), h=tf.concat((forward_state.h, backward_state.h), 1))
        prev_input = tf.concat((forward_output, backward_output), 1)

    #decoder LSTM with dropout
    prev_input = decoder_embedding
    for i in range(num_decoder_layers):
        lstm_dec = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(2 * nodes), output_keep_prob=keep_rate[0], state_keep_prob=keep_rate[0])
        prev_input, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=prev_input, initial_state=last_state, scope='decoder_rnn{0}'.format(str(i)))

    #Dense layers used as Bag of Words
    logits = tf.layers.dense(prev_input, units=translated_vocab_length, use_bias=True)

    #Get the loss and optimize using RMSProp
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, (decoder_size[0] - 1)]))
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

    return (optimizer, loss, logits, inputs, outputs, targets, keep_rate, input_embedding, output_embedding, encoder_embedding, decoder_embedding, decoder_size)

def train_and_save(encoder_input_data, decoder_input_data, optimizer, loss, logits, keep_rate, epochs, batch_size, inputs, outputs, targets, decoder_size, session, modelDir, modelFileName, saver):
    session.run(tf.global_variables_initializer())
    iterations = 100

    #training for some specified number of iterations
    for iteration_i in range(iterations):
        print("iteration: {0}".format(str(iteration_i)))

        #each iteration has some specified number of epochs
        for epoch_i in range(epochs):
            #randomly sample batches from the data
            batch_idx = np.random.choice(np.arange(encoder_input_data.shape[0]), size=batch_size)
            batch_x, batch_y = encoder_input_data[batch_idx, :], decoder_input_data[batch_idx,]

            #iterate over the batches
            for batch_i, (source_batch, target_batch) in enumerate([(batch_x, batch_y)]):
                #removes extra padding in each batch
                source_batch = (source_batch[:, min([np.where(np.array(batch) == 2)[0][0] for batch in source_batch]):])
                target_batch = (target_batch[:, :(max([np.where(np.array(batch) == 2)[0][0] for batch in target_batch]) + 1)])

                #performs training
                _, batch_loss, batch_logits = session.run([optimizer, loss, logits], feed_dict={inputs:source_batch, outputs:target_batch[:, :-1], targets:target_batch[:, 1:], keep_rate:[0.8], decoder_size:[target_batch.shape[1]]})
            accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:, 1:])
            print('\tEpoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f}'.format(epoch_i, batch_loss, accuracy))

        #saves the model
        if (not(iteration_i % 10)):
            try:
                saver.save(session, (modelDir + str(iteration_i) + modelFileName))
                save_path = saver.save(session, (modelDir + modelFileName))
            except:
                os.makedirs(modelDir)
                saver.save(session, (modelDir + str(iteration_i) + modelFileName))
                save_path = saver.save(session, (modelDir + modelFileName))

        print("save path: {0}".format(save_path))

#this is a shortened version of convert_text and is used for the translation session
def prep_test_data(text, original_DNS, max_length):
    temp_text = list()
    temp_text.append(original_DNS['forward']['<GO>'])

    #first clean the text data then convert it to numeric format
    for word in filter_symbols_test(text):
        try: temp_text.append(original_DNS['forward'][word])
        except: temp_text.append(original_DNS['forward']['<UNK>'])

    temp_text.append(original_DNS['forward']['<EOS>'])

    #this truncated, pads, and flips the input text
    temp_text = temp_text[:max_length]
    temp_text[-1] = original_DNS['forward']['<EOS>']
    temp_text = list(reversed(temp_text))
    if len(temp_text) < max_length: temp_text[0:0] = [original_DNS['forward']['<PAD>']]*(max_length-len(temp_text))
    return temp_text

def test(original, translation, original_DNS, translated_DNS, max_original_length, max_translated_length, nodes, embed_size, batch_size, modelDir, modelFileName, isTranslate):
    optimizer, loss, logits, inputs, outputs, targets, keep_rate, input_embedding, output_embedding, encoder_embedding, decoder_embedding, decoder_size = build_model(max_original_length, max_translated_length, len(original_DNS['forward']), len(translated_DNS['forward']), embed_size, nodes, batch_size)
    saver = tf.train.Saver()

    #Use the GPU and some items may not be able to run on the GPU, so use soft placement
    with tf.device('/device:GPU:0'), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        saver.restore(session, (modelDir + modelFileName))
        sfun = SmoothingFunction()
        results = list()

        #This is for testing on a dataset
        if (not(isTranslate)):
            for z, (x, y) in enumerate(list(zip(original, translation))):
                #dec_input will be iteratively updated with each prediction and feb back into the model
                dec_input = np.array([])

                #first token is the <GO> token
                dec_input = np.append(dec_input, translated_DNS['forward']['<GO>'])

                #Iterate for the max number of tokens
                for i in range(1, max_translated_length):
                    #input the parameters into the model and get a result
                    batch_logits = session.run(logits, feed_dict={inputs: [np.trim_zeros(x)], outputs: [dec_input], keep_rate:[1.0], decoder_size:[dec_input.shape[0]]})
                    #append the result to dec_inputs
                    dec_input = np.append(dec_input, (batch_logits[:, -1].argmax(axis=-1)[0]))
                    #if the prediction was <EOS>...break out of loop
                    if translated_DNS['backward'][dec_input[i]] == '<EOS>': break

                #convert from numbers -> words and compute smoothed BLEU score
                input_ = [original_DNS['backward'][k] for k in np.trim_zeros(x)]
                predicted = [translated_DNS['backward'][k] for k in dec_input]
                actual = [translated_DNS['backward'][k] for k in np.trim_zeros(y)]
                results.append((input_, actual, predicted, (nltk.translate.bleu_score.sentence_bleu([actual], predicted, smoothing_function=sfun.method1) * 100)))
                print("{0}/{1}...BLEU Score: {2}".format(z, len(translation), results[-1][-1]))
            #print the mean BLEU score
            print(np.mean([x[-1] for x in results]))
        else:
            #this is for using it to translate a given input in an interactive session
            with open((modelDir + 'parameters.txt'), 'rb') as file: params = pickle.loads(file.read())
            while(1):
                #get input from console, prepare the input for translateion, then translate it
                original = [prep_test_data(input("> "), params['original_DNS'], params['max_original_length'])]
                #the rest here is the same as for testing
                for z, (x, y) in enumerate(list(zip(original, translation))):
                    dec_input = np.array([])
                    dec_input = np.append(dec_input, translated_DNS['forward']['<GO>'])

                    for i in range(1, max_translated_length):
                        batch_logits = session.run(logits, feed_dict={inputs: [np.trim_zeros(x)], outputs: [dec_input], keep_rate: [1.0], decoder_size: [dec_input.shape[0]]})
                        dec_input = np.append(dec_input, (batch_logits[:, -1].argmax(axis=-1)[0]))
                        if translated_DNS['backward'][dec_input[i]] == '<EOS>': break

                    print(' '.join([translated_DNS['backward'][k] for k in dec_input][1:-1]))

#used to load the stored information required for the test function
def testWrapper():
    with open((modelDir + 'parameters.txt'), 'rb') as file: params = pickle.loads(file.read())
    processed_original_test, processed_translated_test = clean_data(test_original, test_translated)
    encoder_input_data_test, decoder_input_data_test = convert_text_test(processed_original_test, processed_translated_test, params['original_DNS'], params['translated_DNS'], params['max_original_length'], params['max_translated_length'])
    test(encoder_input_data_test, decoder_input_data_test, params['original_DNS'], params['translated_DNS'], params['max_original_length'], params['max_translated_length'], params['nodes'], params['embed_size'], params['batch_size'], modelDir, modelFileName, 0)

#trains the model
def train(modelDir, modelFileName):
    epochs = 100
    batch_size = 1
    nodes = 256
    embed_size = 300

    ##loads and cleans the training data from file path names
    processed_original, processed_translated = clean_data(train_original, train_translated)

    #use soft placement on the GPU
    with tf.device('/device:GPU:0'), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        #convert the data to numeric representation
        original_DNS, translated_DNS, encoder_input_data, decoder_input_data, max_original_length, max_translated_length, min_length, unk_original_limit, unk_translated_limit = convert_text(processed_original, processed_translated, 15)
        encoder_input_data = np.array([x for x in encoder_input_data])
        decoder_input_data = np.array([np.array(x) for x in decoder_input_data])

        #builds the model used for training/inference
        optimizer, loss, logits, inputs, outputs, targets, keep_rate, input_embedding, output_embedding, encoder_embedding, decoder_embedding, decoder_size = build_model(max_original_length, max_translated_length, len(original_DNS['forward']), len(translated_DNS['forward']), embed_size, nodes, batch_size)

        #saves the parameters for this session and creates a saver object
        saver = tf.train.Saver()
        with open((modelDir + 'parameters.txt'), 'wb') as file: pickle.dump({'original_DNS': original_DNS, 'translated_DNS': translated_DNS, 'max_original_length': max_original_length, 'max_translated_length': max_translated_length, 'min_length': min_length, 'batch_size': batch_size, 'nodes': nodes, 'embed_size': embed_size}, file)
        #trains and saves the model
        train_and_save(encoder_input_data, decoder_input_data, optimizer, loss, logits, keep_rate, epochs, batch_size, inputs, outputs, targets, decoder_size, session, modelDir, modelFileName, saver)

def trainWrapper():
    train(modelDir, modelFileName)

#loads data from the interactive translation session and calls the test function
def translate():
    with open((modelDir + 'parameters.txt'), 'rb') as file: params = pickle.loads(file.read())
    test(None, [[None]], params['original_DNS'], params['translated_DNS'], params['max_original_length'], params['max_translated_length'], params['nodes'], params['embed_size'], params['batch_size'], modelDir, modelFileName, 1)

if ((__name__ == '__main__') and (len(sys.argv) > 1)):
    code = {'train': 0, 'test': 1, 'translate': 2}
    {0 : trainWrapper, 1: testWrapper, 2: translate}[code[sys.argv[1]]]()
