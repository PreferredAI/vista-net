import tensorflow as tf
import numpy as np
from model_utils import get_shape

try:
  from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
  LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


def bidirectional_rnn(cell_fw, cell_bw, inputs, input_lengths,
                      initial_state_fw=None, initial_state_bw=None,
                      scope=None):
  with tf.variable_scope(scope or 'bi_rnn') as scope:
    (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=cell_fw,
      cell_bw=cell_bw,
      inputs=inputs,
      sequence_length=input_lengths,
      initial_state_fw=initial_state_fw,
      initial_state_bw=initial_state_bw,
      dtype=tf.float32,
      scope=scope
    )
    outputs = tf.concat((fw_outputs, bw_outputs), axis=2)

    def concatenate_state(fw_state, bw_state):
      if isinstance(fw_state, LSTMStateTuple):
        state_c = tf.concat(
          (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
        state_h = tf.concat(
          (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
        state = LSTMStateTuple(c=state_c, h=state_h)
        return state
      elif isinstance(fw_state, tf.Tensor):
        state = tf.concat((fw_state, bw_state), 1,
                          name='bidirectional_concat')
        return state
      elif (isinstance(fw_state, tuple) and
            isinstance(bw_state, tuple) and
            len(fw_state) == len(bw_state)):
        # multilayer
        state = tuple(concatenate_state(fw, bw)
                      for fw, bw in zip(fw_state, bw_state))
        return state

      else:
        raise ValueError(
          'unknown state type: {}'.format((fw_state, bw_state)))

    state = concatenate_state(fw_state, bw_state)
    return outputs, state


def mask_score(scores, sequence_lengths, score_mask_value=tf.constant(-1e15)):
  score_mask = tf.sequence_mask(sequence_lengths, maxlen=tf.shape(scores)[1])
  score_mask_values = score_mask_value * tf.ones_like(scores)
  return tf.where(score_mask, scores, score_mask_values)


def text_attention(inputs,
                   att_dim,
                   sequence_lengths,
                   scope=None):
  assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

  D_w = get_shape(inputs)[-1]
  N_w = get_shape(inputs)[-2]

  with tf.variable_scope(scope or 'text_attention'):
    W = tf.get_variable('W', shape=[D_w, att_dim])
    b = tf.get_variable('b', shape=[att_dim])
    input_proj = tf.nn.tanh(tf.matmul(tf.reshape(inputs, [-1, D_w]), W) + b)

    word_att_W = tf.get_variable(name='word_att_W', shape=[att_dim, 1])
    alpha = tf.matmul(input_proj, word_att_W)
    alpha = tf.reshape(alpha, shape=[-1, N_w])
    alpha = mask_score(alpha, sequence_lengths, tf.constant(-1e15, dtype=tf.float32))
    alpha = tf.nn.softmax(alpha)

    outputs = tf.reduce_sum(inputs * tf.expand_dims(alpha, 2), axis=1)
    return outputs, alpha


def visual_aspect_attention(text_input,  # (b, n_s, d_text)
                            visual_input,  # (b, n_i, d)
                            att_dim,  # d
                            sequence_lengths,
                            scope='visual_aspect_attention'):
  assert len(text_input.get_shape()) == 3 and text_input.get_shape()[-1].value is not None
  assert len(visual_input.get_shape()) == 3 and visual_input.get_shape()[-1].value is not None

  D_t = get_shape(text_input)[-1]
  N_s = get_shape(text_input)[-2]
  D_i = get_shape(visual_input)[-1]
  N_i = get_shape(visual_input)[-2]

  with tf.variable_scope(scope):
    # Sentence-level attention
    W_s = tf.get_variable('W_s', shape=[D_t, att_dim])
    b_s = tf.get_variable('b_s', shape=[att_dim])
    text_input = tf.reshape(text_input, [-1, D_t])
    q = tf.nn.tanh(tf.matmul(text_input, W_s) + b_s)
    q = tf.reshape(q, [-1, 1, N_s, att_dim])

    W_i = tf.get_variable('W_i', shape=[D_i, att_dim])
    b_i = tf.get_variable('b_i', shape=[att_dim])
    visual_input = tf.reshape(visual_input, [-1, D_i])
    p = tf.nn.tanh(tf.matmul(visual_input, W_i) + b_i)
    p = tf.reshape(p, [-1, N_i, 1, att_dim])

    context = tf.multiply(q, p) + q
    context = tf.reshape(context, [-1, att_dim])

    sent_att_W = tf.get_variable(name='sent_att_W', shape=[att_dim, 1])

    beta = tf.matmul(context, sent_att_W)
    beta = tf.reshape(beta, shape=[-1, N_s])

    sequence_lengths = tf.tile(tf.expand_dims(sequence_lengths, axis=1), [1, N_i])
    sequence_lengths = tf.reshape(sequence_lengths, [-1])
    beta = mask_score(beta, sequence_lengths, tf.constant(-1e15, dtype=tf.float32))
    beta = tf.nn.softmax(beta)

    beta = tf.reshape(beta, [-1, N_i, N_s, 1])
    text_input = tf.reshape(text_input, [-1, 1, N_s, D_t])
    weighted_docs = tf.reduce_sum(text_input * beta, axis=2)  # (b, n_i, d)

    # Document-level attention
    W_d = tf.get_variable(name='W_d', shape=[D_t, att_dim])
    b_d = tf.get_variable(name='b_d', shape=[1])
    weighted_docs = tf.reshape(weighted_docs, [-1, D_t])
    doc_proj = tf.nn.tanh(tf.matmul(weighted_docs, W_d) + b_d)

    doc_att_W = tf.get_variable(name='doc_att_W', shape=[att_dim, 1])

    gamma = tf.matmul(doc_proj, doc_att_W)
    gamma = tf.reshape(gamma, shape=[-1, N_i])
    gamma = tf.nn.softmax(gamma)

    weighted_docs = tf.reshape(weighted_docs, [-1, N_i, D_t])
    final_outputs = tf.reduce_sum(weighted_docs * tf.expand_dims(gamma, 2), axis=1)  # (b, d)

    return final_outputs, beta, gamma
