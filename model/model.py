import tensorflow as tf

from model.ops import UNet_block


def generator(inputs, name, reuse=False):
    '''
    Make generator
    :param inputs: inputs
    :param name: scope name
    :param reuse: to reuse or not
    :return: generated mask
    '''
    with tf.variable_scope(name_or_scope=name, default_name=name, reuse=reuse):
        _, g_mask = UNet_block(inputs=inputs, name='g_unet')
        return g_mask


def discriminator(inputs, name, reuse=False):
    '''
    Make discriminator
    :param inputs: inputs
    :param name: scope name
    :param reuse: to reuse or not
    :return: score & predicted mask
    '''
    with tf.variable_scope(name_or_scope=name, default_name=name, reuse=reuse):
        score, d_mask = UNet_block(inputs=inputs, name='d_unet')
        return score, d_mask
