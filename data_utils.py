import os
import numpy as np

photo_feature_dir = 'data/photo_features'
photo_features = {}

MEAN = np.load(os.path.join(photo_feature_dir, 'mean.npy'))


def load_photo(photo_id):
  # global photo_features
  if photo_id not in photo_features.keys():
    photo_feature_path = os.path.join(photo_feature_dir, photo_id[:2], photo_id) + '.npy'
    photo_features[photo_id] = np.load(photo_feature_path)
  return photo_features[photo_id]


def batch_review_normalize(docs):
  batch_size = len(docs)
  document_sizes = np.array([len(doc) for doc in docs], dtype=np.int32)
  document_size = document_sizes.max()
  sentence_sizes_ = [[len(sent) for sent in doc] for doc in docs]
  sentence_size = max(map(max, sentence_sizes_))

  norm_docs = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32)  # == PAD
  sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
  for i, document in enumerate(docs):
    for j, sentence in enumerate(document):
      sentence_sizes[i, j] = sentence_sizes_[i][j]
      for k, word in enumerate(sentence):
        norm_docs[i, j, k] = word

  return norm_docs, document_sizes, sentence_sizes, document_size, sentence_size


def batch_image_normalize(batch_images, num_images):
  batch_size = len(batch_images)

  norm_batch = np.ones(shape=[batch_size, num_images + 1, 4096], dtype=np.float32)
  norm_batch = norm_batch * MEAN

  for i, review_images in enumerate(batch_images):
    for j, image_id in enumerate(review_images):
      norm_batch[i, j, :] = load_photo(image_id)
  return norm_batch