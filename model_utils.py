import tensorflow as tf
import numpy as np
import os

def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims


def make_dirs(dir):
  next_idx = 1
  if tf.gfile.Exists(dir):
    folder_list = sorted([int(f) for f in os.listdir(dir)])
    if len(folder_list) > 0:
      next_idx = folder_list[-1] + 1
  new_folder = os.path.join(dir, str(next_idx))
  tf.gfile.MakeDirs(new_folder)
  return new_folder


def count_parameters(trained_vars):
  total_parameters = 0
  print('=' * 100)
  for variable in trained_vars:
    variable_parameters = 1
    for dim in variable.get_shape():
      variable_parameters *= dim.value
    print('{:70} {:20} params'.format(variable.name, variable_parameters))
    print('-' * 100)
    total_parameters += variable_parameters
  print('=' * 100)
  print("Total trainable parameters: %d" % total_parameters)
  print('=' * 100)


def load_glove(vocab_size, emb_size):
  print('Loading Glove word embeddings ...')
  embedding_weights = {}
  f = open('glove/glove.6B.{}d.txt'.format(emb_size), encoding='utf-8')
  for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_weights[word] = coefs
  f.close()
  print('Total {} word vectors in Glove 6B {}d.'.format(len(embedding_weights), emb_size))

  embedding_matrix = np.random.uniform(-0.5, 0.5, (vocab_size, emb_size))
  embedding_matrix[0, :] = np.zeros(emb_size)  # alignment word for blank image

  oov_count = 0
  from data_preprocess import load_vocabulary
  for word, i in load_vocabulary().items():
    embedding_vector = embedding_weights.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
    else:
      oov_count += 1
  print('Number of OOV words: %d' % oov_count)

  return embedding_matrix
