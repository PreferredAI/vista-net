import os
import gpu_utils

gpu_utils.setup_one_gpu()

import tensorflow as tf
from datetime import datetime

from data_reader import DataReader
from data_preprocess import cities

from model import VistaNet
from model_utils import count_parameters

# Parameters
# ==================================================
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_dir", 'checkpoints',
                       """Path to checkpoint folder""")
tf.flags.DEFINE_string("log_dir", 'log',
                       """Path to log folder""")

tf.flags.DEFINE_integer("num_checkpoints", 1,
                        """Number of checkpoints to store (default: 1)""")
tf.flags.DEFINE_integer("num_epochs", 20,
                        """Number of training epochs (default: 10)""")
tf.flags.DEFINE_integer("batch_size", 32,
                        """Batch Size (default: 32)""")
tf.flags.DEFINE_integer("display_step", 20,
                        """Display after number of steps (default: 20)""")

tf.flags.DEFINE_float("learning_rate", 0.001,
                      """Learning rate (default: 0.001)""")
tf.flags.DEFINE_float("max_grad_norm", 5.0,
                      """Maximum value for gradient clipping (default: 5.0)""")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5,
                      """Probability of keeping neurons (default: 0.5)""")

tf.flags.DEFINE_integer("hidden_dim", 50,
                        """Hidden dimensions of GRU cell (default: 50)""")
tf.flags.DEFINE_integer("att_dim", 100,
                        """Attention dimensions (default: 100)""")
tf.flags.DEFINE_integer("emb_size", 200,
                        """Word embedding size (default: 200)""")
tf.flags.DEFINE_integer("num_images", 3,
                        """Number of images per review (default: 3)""")
tf.flags.DEFINE_integer("num_classes", 5,
                        """Number of classes of prediction (default: 5)""")

tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        """Allow device soft device placement""")

train_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train')
valid_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/valid')

valid_step = 0

def evaluate(session, dataset, model, loss, accuracy, summary_op=None):
  sum_loss = 0.0
  sum_acc = 0.0
  example_count = 0
  global valid_step

  for reviews, images, labels in dataset:
    feed_dict = model.get_feed_dict(reviews, images, labels)
    _loss, _acc = session.run([loss, accuracy], feed_dict=feed_dict)
    if not summary_op is None:
      _summary = session.run(summary_op, feed_dict=feed_dict)
      valid_summary_writer.add_summary(_summary, global_step=valid_step)
      valid_step += len(labels)

    sum_loss += _loss * len(labels)
    sum_acc += _acc * len(labels)
    example_count += len(labels)

  avg_loss = sum_loss / example_count
  avg_acc = sum_acc / example_count
  return avg_loss, avg_acc


def test(session, data_reader, model, loss, accuracy, epoch, result_file):
  for city in cities:
    test_loss, test_acc = evaluate(session, data_reader.read_test_set(city),
                                   model, loss, accuracy)
    result_file.write('city={},epoch={},loss={:.4f},acc={:.4f}\n'.format(city, epoch, test_loss, test_acc))
  result_file.flush()


def train(session, data_reader, model, train_op, loss, accuracy, summary_op):
  for reviews, images, labels in data_reader.read_train_set(batch_size=FLAGS.batch_size):
    step, _, _loss, _acc = session.run([model.global_step, train_op, loss, accuracy],
                                       feed_dict=model.get_feed_dict(reviews, images, labels,
                                                                     FLAGS.dropout_keep_prob))
    if step % FLAGS.display_step == 0:
      _summary = session.run(summary_op, feed_dict=model.get_feed_dict(reviews, images, labels,
                                                                       dropout_keep_prob=1.0))
      train_summary_writer.add_summary(_summary, global_step=step)


def loss_fn(labels, logits):
  onehot_labels = tf.one_hot(labels, depth=FLAGS.num_classes)
  cross_entropy_loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels,
    logits=logits
  )
  tf.summary.scalar('loss', cross_entropy_loss)
  return cross_entropy_loss


def train_fn(loss, global_step):
  trained_vars = tf.trainable_variables()
  count_parameters(trained_vars)

  # Gradient clipping
  gradients = tf.gradients(loss, trained_vars)
  clipped_grads, global_norm = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)
  tf.summary.scalar('global_grad_norm', global_norm)

  # Add gradients and vars to summary
  # for gradient, var in list(zip(clipped_grads, trained_vars)):
  #   if 'attention' in var.name:
  #     tf.summary.histogram(var.name + '/gradient', gradient)
  #     tf.summary.histogram(var.name, var)

  # Define optimizer
  optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
  train_op = optimizer.apply_gradients(zip(clipped_grads, trained_vars),
                                       name='train_op',
                                       global_step=global_step)
  return train_op


def eval_fn(labels, logits):
  prediction = tf.argmax(logits, axis=-1)
  corrected_pred = tf.equal(prediction, tf.cast(labels, tf.int64))
  accuracy = tf.reduce_mean(tf.cast(corrected_pred, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  return accuracy


def main(_):
  config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  with tf.Session(config=config) as sess:
    print('\n{} Model initializing'.format(datetime.now()))

    model = VistaNet(FLAGS.hidden_dim, FLAGS.att_dim, FLAGS.emb_size, FLAGS.num_images, FLAGS.num_classes)
    loss = loss_fn(model.labels, model.logits)
    train_op = train_fn(loss, model.global_step)
    accuracy = eval_fn(model.labels, model.logits)
    summary_op = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    train_summary_writer.add_graph(sess.graph)
    saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)
    data_reader = DataReader(num_images=FLAGS.num_images)

    print('\n{} Start training'.format(datetime.now()))

    epoch = 0
    best_loss = float('inf')
    while epoch < FLAGS.num_epochs:
      epoch += 1
      print('\n=> Epoch: {}'.format(epoch))

      train(sess, data_reader, model, train_op, loss, accuracy, summary_op)

      print('=> Evaluation')
      print('best_loss={:.4f}'.format(best_loss))
      valid_loss, valid_acc = evaluate(sess, data_reader.read_valid_set(batch_size=FLAGS.batch_size),
                                       model, loss, accuracy, summary_op)
      print('valid_loss={:.4f}, valid_acc={:.4f}'.format(valid_loss, valid_acc))

      if valid_loss < best_loss:
        best_loss = valid_loss
        save_path = os.path.join(FLAGS.checkpoint_dir,
                                 'epoch={}-loss={:.4f}-acc={:.4f}'.format(epoch, valid_loss, valid_acc))
        saver.save(sess, save_path)
        print('Best model saved @ {}'.format(save_path))

        print('=> Testing')
        result_file = open(
          os.path.join(FLAGS.log_dir, 'loss={:.4f},acc={:.4f},epoch={}'.format(valid_loss, valid_acc, epoch)), 'w')
        test(sess, data_reader, model, loss, accuracy, epoch, result_file)

  print("{} Optimization Finished!".format(datetime.now()))


if __name__ == '__main__':
  tf.app.run()
