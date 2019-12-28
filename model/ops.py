import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def conv2d(inputs,
           filters: int,
           kernel_size: int,
           strides: int,
           padding: str,
           do_relu: bool,
           do_in: bool,
           name: str):
    '''
    2-D convolution
    :param inputs: inputs
    :param filters: kernel
    :param kernel_size: kernel size
    :param strides: strides
    :param padding: 'SAME'/'VALID'
    :param do_relu: to do relu or not
    :param do_in: to do instance normalization or not
    :param name: scope name
    :return: convoluted results
    '''
    with tf.variable_scope(name_or_scope=name, default_name=name):
        output = tf.layers.conv2d(inputs=inputs,
                                  filters=filters,
                                  kernel_size=kernel_size,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),

                                  # Some optional restrictions
                                  # bias_initializer=tf.constant_initializer(0.1),
                                  # kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                  # bias_regularizer=tf.contrib.layers.l2_regularizer(0.003),

                                  strides=strides,
                                  padding=padding,
                                  name='conv2d')

        if do_in:
            output = tf.contrib.layers.instance_norm(output, scale=False)

        if do_relu:
            output = tf.nn.leaky_relu(output, alpha=0.1, name='relu')

            # To use tanh
            # output = tf.nn.tanh(output, name='tanh')

        return output


def conv2d_transpose(inputs,
                     filters: int,
                     kernel_size: int,
                     strides: int,
                     padding: str,
                     do_relu: bool,
                     do_in: bool,
                     name: str):
    '''
    2-D transposed convolution
    :param inputs: inputs
    :param filters: kernel
    :param kernel_size: kernel size
    :param strides: strides
    :param padding: 'SAME'/'VALID'
    :param do_relu: to do relu or not
    :param do_in: to do instance normalization or not
    :param name: scopr name
    :return: transposed convoluted result
    '''
    with tf.variable_scope(name_or_scope=name, default_name=name):
        output = tf.layers.conv2d_transpose(inputs=inputs,
                                            filters=filters,
                                            kernel_size=kernel_size,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),

                                            # Some optional initializers
                                            # bias_initializer=tf.constant_initializer(0.1),
                                            # kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                            # bias_regularizer=tf.contrib.layers.l2_regularizer(0.003),

                                            strides=strides,
                                            padding=padding,
                                            name='conv2d_transpose')
        if do_in:
            output = tf.contrib.layers.instance_norm(output, scale=False)

        if do_relu:
            output = tf.nn.leaky_relu(output, alpha=0.1, name='relu')

            # To use tanh
            # output = tf.nn.tanh(output, name='tanh')

        return output


def UNet_block(inputs, name):
    '''
    UNet block
    :param inputs: inputs
    :param name: scope name
    :return: results
    '''
    with tf.variable_scope(name_or_scope=name, default_name=name):
        # =====================================================
        # 4 conv ops
        layer1 = conv2d(inputs,
                        filters=64,
                        kernel_size=3,
                        strides=1,
                        padding='SAME',
                        do_relu=True,
                        do_in=True,
                        name='unet_conv1')

        layer2 = conv2d(layer1,
                        filters=128,
                        kernel_size=3,
                        strides=2,
                        padding='SAME',
                        do_relu=True,
                        do_in=True,
                        name='unet_conv2')

        layer3 = conv2d(layer2,
                        filters=256,
                        kernel_size=3,
                        strides=2,
                        padding='SAME',
                        do_relu=True,
                        do_in=True,
                        name='unet_conv3')

        layer4 = conv2d(layer3,
                        filters=512,
                        kernel_size=3,
                        strides=2,
                        padding='SAME',
                        do_relu=True,
                        do_in=True,
                        name='unet_conv4')
        # =====================================================
        # Encoder branch
        # Directly average layer4 as the prediction if the mask is real
        # avg = tf.reduce_mean(layer4, axis=1)
        # avg = tf.reduce_mean(avg, axis=1)
        # avg = tf.reduce_mean(avg, axis=1)

        # Fully connected layer4 and output an integer as the prediction
        avg = slim.flatten(layer4, scope='flatten')
        avg = slim.fully_connected(avg,
                                   num_outputs=256,
                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01),

                                   # Optional restriction
                                   # normalizer_fn=tf.contrib.layers.instance_norm,

                                   activation_fn=tf.nn.leaky_relu)

        avg = slim.fully_connected(avg,
                                   num_outputs=1,
                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                   activation_fn=tf.nn.leaky_relu)
        # =====================================================
        # Decoder branch
        # 4 deconv ops
        layer5 = conv2d_transpose(layer4,
                                  filters=256,
                                  kernel_size=3,
                                  strides=2,
                                  padding='SAME',
                                  do_relu=True,
                                  do_in=True,
                                  name='unet_deconv1')
        layer5 = tf.concat([layer3, layer5], axis=-1)

        layer6 = conv2d_transpose(layer5,
                                  filters=128,
                                  kernel_size=3,
                                  strides=2,
                                  padding='SAME',
                                  do_relu=True,
                                  do_in=True,
                                  name='unet_deconv2')
        layer6 = tf.concat([layer2, layer6], axis=-1)

        layer7 = conv2d_transpose(layer6,
                                  filters=64,
                                  kernel_size=3,
                                  strides=2,
                                  padding='SAME',
                                  do_relu=True,
                                  do_in=True,
                                  name='unet_deconv3')
        layer7 = tf.concat([layer1, layer7], axis=-1)

        layer8 = conv2d(layer7,
                        filters=1,
                        kernel_size=3,
                        strides=1,
                        padding='SAME',
                        do_relu=False,
                        do_in=False,
                        name='unet_conv_final')
        # =====================================================
        layer8 = tf.nn.sigmoid(layer8)
        return avg, layer8


def gen_random_mask_1ch(image_shape, coverage=0.2):
    '''
    Generate random mask
    :param image_shape: shape of mask
    :param coverage: the coverage of mask(the propotion of white pixels)
    :return: generated random mask
    '''

    # Get image shape
    if len(image_shape) == 4:
        (_, height, width, _) = map(int, image_shape)
    else:
        (_, height, width) = map(int, image_shape)

    mask_1ch = np.zeros([height, width])
    all_pixel = height * width

    start_idx_row = np.random.randint(low=int(0.45 * height), high=int(0.55 * height))
    start_idx_col = np.random.randint(low=int(0.45 * height), high=int(0.55 * height))

    last_idx_row = start_idx_row
    last_idx_col = start_idx_col

    # Random direction
    direction = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    cover_pixel = int(coverage * all_pixel)
    for _ in range(cover_pixel):
        # Initial around the selected pixel as white in case that the mask is too slim
        mask_1ch[last_idx_row, last_idx_col] = 1
        mask_1ch[last_idx_row - 1, last_idx_col] = 1
        mask_1ch[last_idx_row + 1, last_idx_col] = 1
        mask_1ch[last_idx_row, last_idx_col - 1] = 1
        mask_1ch[last_idx_row, last_idx_col + 1] = 1
        mask_1ch[last_idx_row - 1, last_idx_col - 1] = 1
        mask_1ch[last_idx_row - 1, last_idx_col + 1] = 1
        mask_1ch[last_idx_row + 1, last_idx_col - 1] = 1
        mask_1ch[last_idx_row + 1, last_idx_col + 1] = 1

        random_direction = np.random.randint(low=0, high=len(direction))
        last_idx_row += direction[random_direction][0]
        last_idx_col += direction[random_direction][1]

        while not (int(0.1 * height + 1) < last_idx_row < int(0.9 * height - 1)
                   and int(0.1 * width + 1) < last_idx_col < int(0.9 * width - 1)):
            random_direction = np.random.randint(low=0, high=len(direction))
            temp_lir = last_idx_row
            temp_lic = last_idx_col

            last_idx_row += direction[random_direction][0] * 1
            last_idx_col += direction[random_direction][1] * 1

            # Adjust the location of white pixels
            if not (int(0.1 * height + 1) < last_idx_row < int(0.9 * height - 1)
                    and int(0.1 * width + 1) < last_idx_col < int(0.9 * width - 1)):
                last_idx_row = temp_lir
                last_idx_col = temp_lic

    mask_1ch = np.expand_dims(mask_1ch, axis=-1)
    # Blur the mask to make the boundary soft(Optional)
    blur_mask = cv2.GaussianBlur(mask_1ch, (3, 3), 0).astype(np.float32)

    return tf.convert_to_tensor(blur_mask)


def composite_image(foreground, background, mask):
    '''
    Composite image
    :param foreground: foreground
    :param background: background
    :param mask: mask
    :return: composited image
    '''

    def _mask_out(image, mask):
        return tf.multiply(image, mask)

    return _mask_out(foreground, mask) + _mask_out(background, tf.subtract(1., mask))
