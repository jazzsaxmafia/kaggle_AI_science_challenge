#-*- coding: utf-8 -*-
import ipdb
import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import rnn

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

        self.lstm_fw_Q = rnn_cell.BasicLSTMCell(dim_hidden)
        self.lstm_bw_Q = rnn_cell.BasicLSTMCell(dim_hidden)

        self.lstm_fw_A = rnn_cell.BasicLSTMCell(dim_hidden)
        self.lstm_bw_A = rnn_cell.BasicLSTMCell(dim_hidden)

        self.W_Q_emb = tf.Variable(tf.random_uniform([dim_hidden*2, dim_embed], -0.1, 0.1), name='W_Q_emb')
        self.b_Q_emb = tf.Variable(tf.zeros([dim_embed]), name='b_Q_emb')

        self.W_A_emb = tf.Variable(tf.random_uniform([dim_hidden*2, dim_embed], -0.1, 0.1), name='W_A_emb')
        self.b_A_emb = tf.Variable(tf.zeros([dim_embed]), name='b_A_emb')

    def build_model(self):
        margin = tf.constant(0.1)
        question = tf.placeholder(tf.int32, [self.batch_size, self.len_question])
        question_sequence_length = tf.placeholder(tf.int64, [self.batch_size])

        answer_right = tf.placeholder(tf.int32, [self.batch_size, self.len_answer])
        answer_sequence_length_right = tf.placeholder(tf.int64, [self.batch_size])

        answer_wrong = tf.placeholder(tf.int32, [self.n_answers - 1, self.batch_size, self.len_answer])
        answer_sequence_length_wrong = tf.placeholder(tf.int64, [self.n_answers - 1, self.batch_size])

        lstm_state_fw_Q = tf.zeros([self.batch_size, self.lstm_fw_Q.state_size])
        lstm_state_bw_Q = tf.zeros([self.batch_size, self.lstm_bw_Q.state_size])

        lstm_state_fw_A = tf.zeros([self.batch_size, self.lstm_fw_A.state_size])
        lstm_state_bw_A = tf.zeros([self.batch_size, self.lstm_bw_A.state_size])

        with tf.device("/cpu:0"):
            embedded_questions = tf.nn.embedding_lookup(self.Wemb, tf.transpose(question))

        question_brnn_output = rnn.bidirectional_rnn(
                self.lstm_fw_Q,
                self.lstm_bw_Q,
                tf.unpack(embedded_questions),
                sequence_length=question_sequence_length,
                initial_state_fw=lstm_state_fw_Q,
                initial_state_bw=lstm_state_bw_Q,
                )
        #question_pooled_output = tf.reduce_mean( tf.pack(question_brnn_output), 0 )
        question_pooled_output = tf.reduce_sum( tf.pack(question_brnn_output), 0 )
        question_pooled_output = question_pooled_output / tf.expand_dims( tf.to_float(question_sequence_length) + 1e-6, 1)
        question_final_emb = tf.nn.xw_plus_b(
                question_pooled_output,
                self.W_Q_emb,
                self.b_Q_emb
                )
        question_final_emb = tf.nn.l2_normalize(question_final_emb, dim=1, epsilon=1e-7)

        def get_similarity(one_answer, one_answer_sequence):

            tf.get_variable_scope().reuse_variables()
            with tf.device("/cpu:0"):
                embedded_answer = tf.nn.embedding_lookup(self.Wemb, tf.transpose(one_answer))

            answer_output = rnn.bidirectional_rnn(
                    self.lstm_fw_A,
                    self.lstm_bw_A,
                    tf.unpack(embedded_answer),
                    sequence_length=one_answer_sequence,
                    initial_state_fw=lstm_state_fw_A,
                    initial_state_bw=lstm_state_bw_A)

            answer_pooled_output = tf.reduce_sum(tf.pack(answer_output), 0)
            answer_pooled_output = answer_pooled_output / tf.expand_dims( tf.to_float(one_answer_sequence) + 1e-6, 1)

            answer_final_emb= tf.nn.xw_plus_b(
                    answer_pooled_output,
                    self.W_A_emb,
                    self.b_A_emb
                    )
            answer_final_emb = tf.nn.l2_normalize(answer_final_emb, dim=1, epsilon=1e-7)
            similarity_matrix = tf.matmul(question_final_emb, tf.transpose(answer_final_emb))
            one_diagonal = tf.diag([1.] * self.batch_size)

            similarity = tf.reduce_sum(similarity_matrix * one_diagonal)
            return similarity

        similarity_right = get_similarity(answer_right, answer_sequence_length_right)

        similarities_wrong = []
        for i in range(self.n_answers - 1):
            similarity_wrong = get_similarity( answer_wrong[i,:,:], answer_sequence_length_wrong[i,:]) #,:,:] )
            similarities_wrong.append(similarity_wrong)

        similarities_wrong = tf.pack(similarities_wrong)
        loss = tf.reduce_mean(tf.maximum( 0., margin + similarities_wrong - similarity_right ))

        return (
                loss,
                margin,
                question,
                question_sequence_length,
                answer_right,
                answer_sequence_length_right,
                answer_wrong,
                answer_sequence_length_wrong
                )

    def build_similarity_calculator(self):
        question = tf.placeholder(tf.int32, [1, self.len_question])
        question_sequence_length = tf.constant([1], dtype=tf.int64)

        answer = tf.placeholder(tf.int32, [1, self.len_answer])
        answer_sequence_length = tf.constant([1], dtype=tf.int64)

        lstm_state_fw_Q = tf.zeros([1, self.lstm_fw_Q.state_size])
        lstm_state_bw_Q = tf.zeros([1, self.lstm_bw_Q.state_size])

        lstm_state_fw_A = tf.zeros([1, self.lstm_fw_A.state_size])
        lstm_state_bw_A = tf.zeros([1, self.lstm_bw_A.state_size])

        with tf.device("/cpu:0"):
            embedded_question = tf.nn.embedding_lookup(self.Wemb, tf.transpose(question))
            embedded_answer = tf.nn.embedding_lookup(self.Wemb, tf.transpose(answer))

        question_brnn_output = rnn.bidirectional_rnn(
                self.lstm_fw_Q,
                self.lstm_bw_Q,
                tf.unpack(embedded_question),
                sequence_length=question_sequence_length,
                initial_state_fw=lstm_state_fw_Q,
                initial_state_bw=lstm_state_bw_Q,
                )
        question_pooled_output = tf.reduce_mean( tf.pack(question_brnn_output), 0 )
        question_final_emb = tf.nn.xw_plus_b(
                question_pooled_output,
                self.W_Q_emb,
                self.b_Q_emb
                )
        question_final_emb = tf.nn.l2_normalize(question_final_emb, dim=1)

        answer_brnn_output = rnn.bidirectional_rnn(
                    self.lstm_fw_A,
                    self.lstm_bw_A,
                    tf.unpack(embedded_answer),
                    sequence_length=answer_sequence_length,
                    initial_state_fw=lstm_state_fw_A,
                    initial_state_bw=lstm_state_bw_A)

        answer_brnn_pooled_output = tf.reduce_mean(tf.pack(answer_brnn_output), 0)
        answer_final_emb= tf.nn.xw_plus_b(
                    answer_brnn_pooled_output,
                    self.W_A_emb,
                    self.b_A_emb
                    )
        answer_final_emb = tf.nn.l2_normalize(answer_final_emb, dim=1)

        similarity = tf.reduce_mean(tf.matmul(question_final_emb, tf.transpose(answer_final_emb)))
        return question, question_sequence_length, answer, answer_sequence_length, similarity
