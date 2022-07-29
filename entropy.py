import tensorflow as tf
from tensorflow.keras import datasets
import json
from nasbench_keras import ModelSpec, build_keras_model, build_module
from nasbench_keras import generate_graphs
import numpy as np
from scipy.stats import entropy
from os.path import exists
import time
import argparse


def generate_nb_graphs(filename):
    generate_graphs(output_file=filename, max_vertices=7, num_ops=3, max_edges=9, verify_isomorphism=True)


def augment_sample(img, seed=None):
    img_flip_lr = tf.image.flip_left_right(img).numpy()
    img_flip_ud = tf.image.flip_up_down(img).numpy()
    img_bright = tf.image.random_brightness(img, max_delta=0.5, seed=seed).numpy()
    img_contrast = tf.image.random_contrast(img, lower=0.2, upper=0.5, seed=seed).numpy()
    img_saturation = tf.image.random_saturation(img, lower=5, upper=10, seed=seed).numpy()
    img_jpeg = tf.image.random_jpeg_quality(img, min_jpeg_quality=25, max_jpeg_quality=95, seed=seed).numpy()
    img_hue = tf.image.random_hue(img, max_delta=0.2, seed=seed).numpy()
    return (img,
           img_flip_lr,
           img_flip_ud,
           img_bright,
           img_contrast,
           img_saturation,
           img_jpeg,
           img_hue)


def prepare_weights(net):
    weights = list()
    for w in range(len(net.weights)):
        if net.weights[w].trainable:
            if net.weights[w].name.find('beta') > 0: 
                weights.append(np.random.normal(
                    loc=0.0,
                    scale=1e-2,
                    size=net.weights[w].shape))
            elif net.weights[w].name.find('gamma') > 0: 
                weights.append(np.random.normal(
                    loc=1.0,
                    scale=1e-2,
                    size=net.weights[w].shape))
            else:
                weights.append(np.random.normal(
                    loc=0.0,
                    scale=5e-2,
                    size=net.weights[w].shape))
        else: # non trainable weights
            weights.append(net.weights[w].numpy())
    # we manually set the weights
    return weights


def evaluate_model(
        model,
        config,
        X,
        y,
        samp_per_weights=1,
        seed=None):
    """ Compute metrics
    Input:
        model    NAS-Bench-101 model description (matrix and labels)
        config   
    """
    # Adjacency matrix and nuberically-coded layer list
    matrix, labels = model
    # Transfer numerically-coded operations to layers (check base_ops.py)
    labels = (['input'] + [config['available_ops'][l] for l in labels[1:-1]] + ['output'])
    # Module graph
    spec = ModelSpec(matrix, labels, data_format='channels_last')
    # Create module
    # inputs = tf.keras.layers.Input(train_images.shape[1:], 1)
    # outputs = build_module(spec=spec, inputs=inputs, channels=128, is_training=True)
    # module = tf.keras.Model(inputs=inputs, outputs=outputs)
    # module.summary()
    # Create whole network with same config
    features = tf.keras.layers.Input(train_images.shape[1:], 1)
    net_outputs = build_keras_model(spec, features, labels, config)
    net = tf.keras.Model(inputs=features, outputs=net_outputs)
    # net.summary()
    aug_h = list()
    aug_base_h = list()
    y_prima = dict()
    for k in np.unique(y):
        y_prima[k] = None
    for i in range(len(X)):
        img = X[i] / 255
        # augment data
        augmented_batch = augment_sample(img, seed)
        # prepare the batch
        batch = np.stack(augmented_batch)
        # reset the weights, predict, and compute entropy
        if 1 % samp_per_weights == 0:
            weights = prepare_weights(net)
            net.set_weights(weights)
        _pred = tf.nn.softmax(net.predict(batch)).numpy()
        if y_prima[int(y[i])] is None:
            y_prima[int(y[i])] = [list(_pred[0].copy())]
        else:
            y_prima[int(y[i])].append(list(_pred[0].copy()))
        # to avoid NaN
        _pred[_pred==0.0] = 1e-20
        aug_h.append(np.sum(entropy(_pred, axis=0))) # column entropy, i.e., tolerance to augmentation (high value)
        aug_base_h.append(np.sum(entropy(_pred, axis=1))) # row entropy, i.e., ability to diff classes (low value)
    # compute the entropy index
    aug_index = np.mean(np.array(aug_h) / np.array(aug_base_h))
    #
    pred_h = list()
    pred_base_h = list()
    while True:
        # first, we prepare a batch with one sample from each class
        batch = list()
        for k in y_prima.keys():
            if len(y_prima[k]) == 0:
               break
            batch.append(y_prima[k].pop())
        if len(batch) < len(y_prima.keys()):
            break
        # now, we compute the entropy
        pred_h.append(np.sum(entropy(batch, axis=0))) # column entropy, i.e., a low value means that samples from different classes
                                              # are treated differently
        pred_base_h.append(np.sum(entropy(batch, axis=1))) # row entropy
    # we compute a "normalized" entropy index
    pred_index = np.mean(np.array(pred_h) / np.array(pred_base_h))
    metrics = {
        'aug_h': float(np.mean(aug_h)), 
        'aug_base_h': float(np.mean(aug_base_h)),
        'aug_index': float(aug_index),
        'pred_h': float(np.mean(pred_h)),
        'pred_base_h': float(np.mean(pred_base_h)),
        'pred_index': float(pred_index),
        'matrix': matrix,
        'labels': labels}
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--seed',
      type=int,
      default=None,
      help='Random seed (default None).')
    parser.add_argument(
      '--first',
      type=int,
      default=0,
      help='Initial mode (default 0).')
    parser.add_argument(
      '--last',
      type=int,
      default=1,
      help='Last model (default 1).')
    parser.add_argument(
      '--samples',
      type=int,
      default=100,
      help='Number of samples (default 100).')
    parser.add_argument(
      '--refresh',
      type=int,
      default=10,
      help='Number of samples to evaluate before updating the weights (default 10).')
    parser.add_argument(
      '--graphs',
      type=str,
      default='data/generated_graphs1.json',
      help='File containing the NB-101 graphs (default data/generated_graphs1.json.')
    parser.add_argument(
      '--out',
      type=str,
      default='data/evaluated_models.json',
      help='Output file (default data/evaluated_models.json.')

    flags, unparsed = parser.parse_known_args()
    print("Config", 	flags)

    # load CIFAR10 data
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    if not exists(flags.graphs):
        generate_nb_graphs(flags.graphs)

    with open(flags.graphs, "rb") as f:
        models = json.load(f)

    # Configure whole network based on https://arxiv.org/pdf/1902.09635.pdf
    config = {'available_ops' : ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'],
              'stem_filter_size' : 128, # initial layer is 3x3 conv with 128 out channels
              'data_format' : 'channels_last', # 'channels_last' or 'channels_first' (depends on data)
              'num_stacks' : 3, # 3 stacks, as defined in the original data set
              'num_modules_per_stack' : 3, # 3 cells per stack, as defined in the paper
              'num_labels' : 10} # number of output classes

    # Get model by the hash
    #model = models['0001a2f6c8977346ccd12fa0c435bf42']
    model_keys = list(models.keys())

    for _key in model_keys[flags.first:flags.last]:
        model = models[_key]
        start = time.time()
        metrics = evaluate_model(
            model=model,
            config=config,
            X=train_images[:flags.samples],
            y=train_labels[:flags.samples],
            samp_per_weights=flags.refresh,
            seed=flags.seed)
        metrics['eval_time'] = time.time() - start
        metrics['model_key'] = _key

        with open(flags.out, 'a') as f:
            f.write(json.dumps(metrics) + '\n')


