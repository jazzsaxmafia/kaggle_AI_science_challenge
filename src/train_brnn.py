#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
import ipdb
import os

from model_brnn import Model
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

def get_data(train_data):
    #train_data = pd.read_table(data_path, sep='\t')

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

def get_sequence_length(seq, pad=0):
    nonzeros = np.array( map(lambda x: (x!=0).sum(), seq))
    return nonzeros

#### 잡다한 Parameter ####
log_path = './log.txt'
model_path = '../models'
train_data_path = '../data/training_set.tsv'

data = pd.read_table(train_data_path, sep='\t')

train_data = data[:int(len(data)*0.9)]
valid_data = data[int(len(data)*0.9):]

questions, correct_answer, wrong_answer, all_answers, all_sentences = get_data(train_data)
questions_v, correct_answer_v, wrong_answer_v, all_answers_v, all_sentences_v = get_data(valid_data)

wordtoix, ixtoword, bias_init_vector = build_vocab(all_sentences, 2)

##########################

#### Train Parameters ####
n_epochs = 100
learning_rate = 0.001
margin = 0.1
batch_size = 50
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

(
    loss,
    margin,
    question,
    question_sequence_length,
    answer_right,
    answer_sequence_length_right,
    answer_wrong,
    answer_sequence_length_wrong
) = model.build_model()

question_test, question_length_test, answer_test, answer_length_test, similarity_test = model.build_similarity_calculator()
sess = tf.InteractiveSession()

saver = tf.train.Saver(max_to_keep=1000)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
tf.initialize_all_variables().run()

valid_question_ind = map(lambda q: [wordtoix[word] for word in q.split(' ') if word in wordtoix], questions_v)
valid_answer_ind_correct = map(lambda q: [wordtoix[word] for word in q.split(' ') if word in wordtoix], correct_answer_v)
valid_answer_ind_wrong = map(lambda qs: map(lambda q: [wordtoix[word] for word in q.split(' ') if word in wordtoix], qs), wrong_answer_v)

valid_question_matrix = sequence.pad_sequences(valid_question_ind, padding='post', maxlen=len_question)
valid_answer_matrix_correct = sequence.pad_sequences(valid_answer_ind_correct, padding='post', maxlen=len_answer)
valid_answer_matrix_wrong = [sequence.pad_sequences(x, padding='post', maxlen=len_answer) for x in valid_answer_ind_wrong]

valid_question_length = get_sequence_length(valid_question_matrix)
valid_answer_length_correct = get_sequence_length(valid_answer_matrix_correct)
valid_answer_length_wrong = [get_sequence_length(i) for i in valid_answer_matrix_wrong]

for epoch in range(n_epochs):
    index = range(len(questions))
    np.random.shuffle(index)

    training_loss = 0.
    training_iter = 0.

    for start, end in zip(
        range(0, len(index), batch_size),
        range(batch_size, len(index), batch_size)
        ):

        batch_index = index[start:end]
        batch_question = questions[batch_index]
        batch_correct_answer = correct_answer[batch_index]
        batch_wrong_answer = wrong_answer[batch_index]

        batch_question_ind = map(lambda q: [wordtoix[word] for word in q.split(' ') if word in wordtoix], batch_question)
        batch_answer_ind_correct = map(lambda q: [wordtoix[word] for word in q.split(' ') if word in wordtoix], batch_correct_answer)
        batch_answer_ind_wrong = map(lambda qs: map(lambda q: [wordtoix[word] for word in q.split(' ') if word in wordtoix], qs), batch_wrong_answer)

        batch_question_matrix = sequence.pad_sequences(batch_question_ind, padding='post', maxlen=len_question)
        batch_question_length = get_sequence_length(batch_question_matrix)

        batch_correct_answer_matrix = sequence.pad_sequences(batch_answer_ind_correct, padding='post', maxlen=len_answer)

        batch_wrong_answer_matrix = [sequence.pad_sequences(x, padding='post', maxlen=len_answer) for x in batch_answer_ind_wrong]

        batch_answer_length_correct = get_sequence_length(batch_correct_answer_matrix)
        batch_answer_length_wrong = [get_sequence_length(i) for i in batch_wrong_answer_matrix]

        _, loss_val = sess.run(
                [train_op, loss],
                feed_dict={
                    margin:0.1,
                    question:batch_question_matrix,
                    question_sequence_length:batch_question_length,
                    answer_right:batch_correct_answer_matrix,
                    answer_sequence_length_right:batch_answer_length_correct,
                    answer_wrong: np.array(batch_wrong_answer_matrix).swapaxes(1,0),
                    answer_sequence_length_wrong: np.array(batch_answer_length_wrong).swapaxes(1,0)
                    })

        training_loss += loss_val
        training_iter += 1

        #print loss_val

    print "Epoch ", epoch, " is finished. Training loss: ", training_loss / training_iter
    saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

    valid_score = 0
    for (
            valid_q,
            valid_q_len,
            valid_a_c,
            valid_a_c_len,
            valid_a_ws,
            valid_a_ws_len
        ) in zip(
                valid_question_matrix,
                valid_question_length,
                valid_answer_matrix_correct,
                valid_answer_length_correct,
                valid_answer_matrix_wrong,
                valid_answer_length_wrong
        ):

        right_similarity = sess.run(
                similarity_test,
                feed_dict={
                    question_test:[valid_q],
                    question_length_test:[valid_q_len],
                    answer_test:[valid_a_c],
                    answer_length_test:[valid_a_c_len],
                    })

        wrong_similarities = []
        for (valid_a_w, valid_a_w_len) in zip(valid_a_ws, valid_a_ws_len):
            wrong_similarity = sess.run(
                    similarity_test,
                    feed_dict={
                        question_test:[valid_q],
                        question_length_test:[valid_q_len],
                        answer_test:[valid_a_w],
                        answer_length_test:[valid_a_w_len],
                        })
            wrong_similarities.append(wrong_similarity)

        if np.all(right_similarity - np.array(wrong_similarities) > 0):
            valid_score += 1

    valid_score = float(valid_score) / len(valid_question_matrix)
    print "Validation Score for Epoch ", epoch, " : ", valid_score

    with open(log_path, "a") as f:
        f.write(str(epoch)+'\t'+str(valid_score))
        f.write('\n')







