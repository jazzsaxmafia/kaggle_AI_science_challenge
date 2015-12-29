#-*- coding: utf-8 -*-
import ipdb
import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell

class Model():
    def __init__(self, batch_size, len_question, len_answer, n_answers, n_words, dim_embed, dim_hidden, bias_init_vector=None):

        self.batch_size = batch_size
        self.len_question = len_question
        self.len_answer = len_answer
        self.n_answers = n_answers
        self.n_words = n_words
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -0.1, 0.1), name='Wemb')

        self.W_emb_hid_Q = tf.Variable(tf.random_uniform([dim_embed, dim_hidden], -0.1, 0.1), name='W_emb_hid_Q')
        self.b_emb_hid_Q = tf.Variable(tf.zeros([dim_hidden]), name='b_emb_hid_Q')

        self.W_emb_hid_A = tf.Variable(tf.random_uniform([dim_embed, dim_hidden], -0.1, 0.1), name='W_emb_hid_A')
        self.b_emb_hid_A = tf.Variable(tf.zeros([dim_hidden]), name='b_emb_hid_Q')

        self.lstm1 = rnn_cell.BasicLSTMCell(dim_hidden)
        self.lstm2 = rnn_cell.BasicLSTMCell(dim_hidden)

        self.W_hid_emb = tf.Variable(tf.random_uniform([dim_hidden, dim_embed], -0.1, 0.1), name='W_hid_emb')
        self.b_hid_emb = tf.Variable(tf.zeros([dim_embed]), name='b_hid_emb')

        self.W_emb_word = tf.Variable(tf.random_uniform([dim_embed, n_words], -0.1, 0.1), name='W_emb_word')

        if bias_init_vector is not None:
            self.b_embed_word = tf.Variable(bias_init_vector.astype(np.float32), name='b_embed_word')
        else:
            self.b_emb_word = tf.Variable(tf.zeros([n_words]), name='b_emb_word')

    def build_model(self):
        #margin = tf.placeholder(tf.float32)
        margin = tf.constant(0.1)
        question = tf.placeholder(tf.int32, [self.batch_size, self.len_question])
        question_mask = tf.placeholder(tf.float32, [self.batch_size, self.len_question])

        answer_right = tf.placeholder(tf.int32, [self.batch_size, self.len_answer])
        answer_wrong = tf.placeholder(tf.int32, [self.n_answers - 1, self.batch_size, self.len_answer])
        answer_mask = tf.placeholder(tf.float32, [self.batch_size, self.len_answer])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm1.state_size])

        output1 = tf.zeros([self.batch_size, self.dim_hidden])
        output2 = tf.zeros([self.batch_size, self.dim_hidden])

        padding = tf.zeros([self.batch_size, self.dim_hidden])

        for i in range(self.len_question):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.device("/cpu:0"):
                current_question_embed = tf.nn.embedding_lookup(self.Wemb, question[:,i])
            current_mask = tf.expand_dims( question_mask[:, i], 1 )

            current_question_hidden = tf.nn.xw_plus_b( current_question_embed, self.W_emb_hid_Q, self.b_emb_hid_Q )

            (prev_output1, prev_state1) = (output1, state1)
            (prev_output2, prev_state2) = (output2, state2)

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1( current_question_hidden, state1 )

            output1 = current_mask * output1 + (1. - current_mask) * prev_output1
            state1 = current_mask * state1 + (1. - current_mask) * prev_state1

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2( tf.concat(1, [padding, output1]), state2 )

            output2 = current_mask * output2 + (1. - current_mask) * prev_output2
            state2 = current_mask * state2 + (1. - current_mask) * prev_state2

        encoder_state1 = state1
        encoder_state2 = state2
        encoder_output1 = output1
        encoder_output2 = output2

        def get_loss(one_answer):
            loss = 0.0

            state1 = encoder_state1
            state2 = encoder_state2
            output1 = encoder_output1
            output2 = encoder_output2

            for i in range(self.len_answer):
                if i == 0:
                    current_answer_embed = tf.zeros([self.batch_size, self.dim_embed])
                else:
                    with tf.device("/cpu:0"):
                        current_answer_embed = tf.nn.embedding_lookup(self.Wemb, one_answer[:,i-1])

                tf.get_variable_scope().reuse_variables()

                current_answer_hidden = tf.nn.xw_plus_b(current_answer_embed, self.W_emb_hid_A, self.b_emb_hid_A)

                (prev_output1, prev_state1) = (output1, state1)
                (prev_output2, prev_state2) = (output2, state2)

                current_mask = tf.expand_dims(answer_mask[:,i], 1)

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(padding, state1)

                output1 = current_mask * output1 + (1. - current_mask) * prev_output1
                state1 = current_mask * state1 + (1. - current_mask) * prev_state1

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2( tf.concat(1, [current_answer_hidden, output1]), state2)

                output2 = current_mask * output2 + (1. - current_mask) * prev_output2
                state2 = current_mask * state2 + (1. - current_mask) * prev_state2

                labels = tf.expand_dims( one_answer[:,i], 1)
                indices = tf.expand_dims( tf.range(0, self.batch_size, 1), 1)

                concated = tf.concat(1, [indices, labels])
                onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)

                logit_embed = tf.nn.relu(tf.nn.xw_plus_b(output2, self.W_hid_emb, self.b_hid_emb))
                logit_words = tf.nn.xw_plus_b(logit_embed, self.W_emb_word, self.b_emb_word)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
                cross_entropy = cross_entropy * answer_mask[:,i]

                current_loss = tf.reduce_sum(cross_entropy)
                loss += current_loss

            return loss

        loss_right = get_loss(answer_right)

        losses_wrong = []
        for i in range(self.n_answers - 1):
            loss_wrong = get_loss( answer_wrong[0,:,:] )
            losses_wrong.append(loss_wrong)

        losses_wrong = tf.pack(losses_wrong)
        loss = tf.reduce_mean(tf.maximum( 0., margin + losses_wrong - loss_right ))

        return loss, margin, question, question_mask, answer_right, answer_wrong, answer_mask

