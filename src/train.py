#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ipdb

from model import Model
from keras.preprocessing import sequence

def get_correct_answer(row):

    answer_columns = ['answerA', 'answerB', 'answerC', 'answerD']
    all_answers = row[answer_columns]
    correct_answer_index = 'answer'+row['correctAnswer']

    correct_answer = row[correct_answer_index].lower()
    wrong_answers = all_answers.drop(correct_answer_index).values
    wrong_answers = [wrong_answer.lower() for wrong_answer in wrong_answers]

    return correct_answer, wrong_answers

def build_vocab(sentence_iterator, word_count_threshold=1):
    # borrowed this function from NeuralTalk
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector

def split_special_chars(sent):
    special_characters = ["?", "\"", "\'", "/", "!", "(", ")", ".", ", "]
    for char in special_characters:
        words = sent.lower().split(char)
        words = filter(lambda x: x != ' ', words)
        sent = ' '.join(words)
        #sent =  ' '.join(sent.lower().split(char))

    return sent

def get_data(data_path):
    train_data = pd.read_table(data_path, sep='\t')

    questions = train_data['question'].values
    questions = np.array(map(lambda x: split_special_chars(x), questions))

    answer_tuple = train_data.apply(lambda row: get_correct_answer(row), axis=1)
    correct_answer = answer_tuple.map(lambda x: x[0])
    correct_answer = np.array(map(lambda x: split_special_chars(x), correct_answer))

    wrong_answers = answer_tuple.map(lambda x: x[1])
    wrong_answers = np.array(map(lambda x: [split_special_chars(y) for y in x], wrong_answers))

    all_answers = np.hstack([correct_answer, np.hstack(wrong_answers)] )
    all_sentences = np.hstack( [questions, all_answers] )

    return questions, correct_answer, wrong_answers, all_answers, all_sentences

#### 잡다한 Parameter ####
train_data_path = '../data/training_set.tsv'
questions, correct_answer, wrong_answer, all_answers, all_sentences = get_data(train_data_path)
wordtoix, ixtoword, bias_init_vector = build_vocab(all_sentences)
##########################

#### Train Parameters ####
n_epochs = 100
learning_rate = 0.001

batch_size = 30
len_question = np.max(map(lambda x: len(x.split(' ')), questions))
len_answer = np.max(map(lambda x: len(x.split(' ')), all_answers))
n_answers=4
n_words = len(wordtoix)
dim_embed = 256
dim_hidden = 256
##########################

model = Model(
        batch_size=batch_size,
        len_question=len_question,
        len_answer=len_answer,
        n_answers=n_answers,
        n_words=n_words,
        dim_embed=dim_embed,
        dim_hidden=dim_hidden
        )

loss, margin, question, question_mask, answer_right, answer_wrong, answer_mask = model.build_model()


for epoch in range(n_epochs):
    index = range(len(questions))
    np.random.shuffle(index)

    for start, end in zip(
        range(0, len(index), batch_size),
        range(batch_size, len(index), batch_size)
        ):

        batch_index = index[start:end]
        batch_question = questions[batch_index]
        batch_correct_answer = correct_answer[batch_index]
        batch_wrong_answer = wrong_answer[batch_index]

        batch_question_ind = map(lambda q: [wordtoix[word] for word in q.split(' ') if word in wordtoix], batch_question)
        batch_correct_answer_ind = map(lambda q: [wordtoix[word] for word in q.split(' ') if word in wordtoix], batch_correct_answer)
        batch_wrong_answer_ind = map(lambda qs: map(lambda q: [wordtoix[word] for word in q.split(' ') if word in wordtoix], qs), batch_wrong_answer)

        batch_question_matrix = sequence.pad_sequences(batch_question_ind, padding='post', maxlen=len_question)
        batch_correct_answer_matrix = sequence.pad_sequences(batch_correct_answer_ind, padding='post', maxlen=len_answer)
        batch_wrong_answer_matrix = [sequence.pad_sequences(x, padding='post', maxlen=len_answer) for x in batch_wrong_answer_ind]
        batch_wrong_answer_matrix = np.array(batch_wrong_answer_matrix).swapaxes(0,1)
