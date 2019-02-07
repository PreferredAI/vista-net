import json
import os
import pickle
import sys
from collections import defaultdict
from tqdm import tqdm

cities = ['Boston', 'Chicago', 'Los Angeles', 'New York', 'San Francisco']

data_dir = 'data'

train_raw_file = os.path.join(data_dir, 'train.json')
valid_raw_file = os.path.join(data_dir, 'valid.json')

train_file = os.path.join(data_dir, 'train.pickle')
valid_file = os.path.join(data_dir, 'valid.pickle')
vocab_file = os.path.join(data_dir, 'vocab.pickle')
word_freq_file = os.path.join(data_dir, 'word-freq.pickle')

VOCAB_SIZE = 40000
UNK = 2
SENT_DELIMITER = '|||'


def read_reviews(file_path):
  reviews = []
  with open(file_path, 'r') as f:
    for line in tqdm(f):
      review = json.loads(line)
      photos = []
      for photo in review['Photos']:
        photos.append(photo['_id'])
      reviews.append({'_id': review['_id'],
                      'Text': review['Text'],
                      'Photos': photos,
                      'Rating': review['Rating']})
  return reviews


def word_tokenize(text):
  for sent in text.split(SENT_DELIMITER):
    for word in sent.split():
      yield word


def build_word_freq():
  try:
    with open(word_freq_file, 'rb') as freq_dist_f:
      freq_dist_f = pickle.load(freq_dist_f)
      print('word frequency loaded')
      return freq_dist_f
  except IOError:
    pass

  print('building word frequency')
  word_freq = defaultdict(int)
  # from train file
  for i, review in enumerate(read_reviews(train_raw_file)):
    for word in word_tokenize(review['Text']):
      word_freq[word] += 1
  # from validation file
  for i, review in enumerate(read_reviews(valid_raw_file)):
    for word in word_tokenize(review['Text']):
      word_freq[word] += 1
  with open(word_freq_file, 'wb') as f:
    pickle.dump(word_freq, f)
  return word_freq


def build_vocabulary():
  print('building vocabulary')
  word_freq = build_word_freq()
  top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:VOCAB_SIZE - 3]
  print('most common word is %s which appears %d times' % (top_words[0][0], top_words[0][1]))
  print('less common word is %s which appears %d times' % (top_words[-1][0], top_words[-1][1]))
  vocab = {}
  i = 3  # 0-index is for padding, 2-index is for UNKNOWN word
  for word, freq in top_words:
    vocab[word] = i
    i += 1
  with open(vocab_file, 'wb') as f:
    pickle.dump(vocab, f)


def load_vocabulary():
  try:
    with open(vocab_file, 'rb') as f:
      vocab = pickle.load(f)
      print('Vocabulary loaded')
      return vocab
  except IOError:
    print('Can not load vocabulary')
    sys.exit(0)


def dump_file(input_file, output_file):
  if os.path.exists(output_file):
    print('%s is dumped already' % output_file)
    return

  vocab = load_vocabulary()
  print('start dumping %s into %s' % (input_file, output_file))
  f = open(output_file, 'wb')
  try:
    for review in read_reviews(input_file):
      rating = review['Rating']
      photos = review['Photos']
      text = []
      for sent in review['Text'].split(SENT_DELIMITER):
        text.append([vocab.get(word, UNK) for word in sent.split()])
      pickle.dump((text, photos, rating), f)
  except KeyboardInterrupt:
    pass
  f.close()


if __name__ == '__main__':
  build_vocabulary()

  dump_file(train_raw_file, train_file)
  dump_file(valid_raw_file, valid_file)
  for city in cities:
    dump_file(os.path.join(data_dir, 'test/{}_test.json'.format(city)),
              os.path.join(data_dir, 'test/{}_test.pickle'.format(city)))
