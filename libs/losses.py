import tensorflow as tf


def mask_loss(model_mask, mask):
    '''
    Calculate loss for generated masks(symmetric for binary masks)
    :param model_mask: pred mask
    :param mask: gt mask
    :return: mask loss
    '''
    return tf.reduce_min((cross_entropy_loss(model_mask, mask),
                          cross_entropy_loss(model_mask, 1. - mask)), axis=0)


def cross_entropy_loss(logits, labels):
    '''
    CE loss
    :param logits: logits
    :param labels: labels
    :return: CE loss
    '''

    def plogq(p, q):
        return tf.multiply(p, tf.log(q + 1e-10))

    # If labels are not the same shape as logits
    if type(labels) == float:
        labels = tf.fill(dims=logits.shape, value=labels)

    return tf.reduce_mean(-plogq(labels, logits) - plogq(1. - labels, 1. - logits))
