import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glob import glob

import cv2

import numpy as np
import tensorflow as tf

import cfgs
from cfgs import logger
from model.model import generator, discriminator
from model.ops import composite_image, gen_random_mask_1ch
from tqdm import tqdm


def main():
    '''
    Main function
    '''
    with tf.variable_scope(name_or_scope='cp_gan') as scope:
        # ___________________________Preparation Work______________________________________
        foreground_plh = tf.placeholder(dtype=tf.float32,
                                        shape=[cfgs.IMG_HEIGHT, cfgs.IMG_WIDTH, cfgs.CHANNEL])
        background_plh = tf.placeholder(dtype=tf.float32,
                                        shape=[cfgs.IMG_HEIGHT, cfgs.IMG_WIDTH, cfgs.CHANNEL])
        shuffled_foreground_plh = tf.placeholder(dtype=tf.float32,
                                                 shape=[cfgs.IMG_HEIGHT, cfgs.IMG_WIDTH, cfgs.CHANNEL])

        # ___________________________Image Preprocessing____________________________________
        foreground = random_process_image(foreground_plh)
        background = random_process_image(background_plh)
        shuffled_foreground = random_process_image(shuffled_foreground_plh)

        foreground = tf.expand_dims(foreground, axis=0)
        background = tf.expand_dims(background, axis=0)
        shuffled_foreground = tf.expand_dims(shuffled_foreground, axis=0)

        # ___________________________Create the graph, etc__________________________________
        g_mask_foreground = generator(inputs=foreground, name='generator')
        score_foreground, d_mask_foreground = discriminator(inputs=foreground, name='discriminator')

        composited_image = composite_image(foreground, background, g_mask_foreground)
        anti_shortcut_image = composite_image(foreground=shuffled_foreground,
                                              background=background,
                                              mask=g_mask_foreground)

        random_mask = tf.py_function(func=gen_random_mask_1ch, inp=[foreground.shape, 0.2], Tout=[tf.float32])
        random_mask = tf.stack(values=[random_mask, random_mask, random_mask], axis=-1)
        random_mask.set_shape(foreground.shape)

        grounded_fake_image = composite_image(foreground=foreground,
                                              background=background,
                                              mask=random_mask)

        scope.reuse_variables()
        score_composite, d_mask_composite = discriminator(inputs=composited_image, name='discriminator', reuse=True)

        scope.reuse_variables()
        score_anti_shortcut, d_mask_anti_shortcut = discriminator(inputs=anti_shortcut_image, name='discriminator',
                                                                  reuse=True)
        scope.reuse_variables()
        score_grounded_fake, d_mask_grounded_fake = discriminator(inputs=grounded_fake_image, name='discriminator',
                                                                  reuse=True)

        # _____________________________Define Losses_________________________________
        # Score losses
        # d_real = cross_entropy_loss(logits=score_foreground, labels=1.)
        # d_fake = cross_entropy_loss(logits=score_composite, labels=0.)

        # g_fake = cross_entropy_loss(logits=score_composite, labels=1.)
        # g_anti_shortcut = cross_entropy_loss(logits=score_anti_shortcut, labels=0.)

        # d_grounded_fake = cross_entropy_loss(logits=score_grounded_fake, labels=0.)

        # Mask losses
        # d_mask_real = mask_loss(model_mask=d_mask_foreground, mask=0.)
        # d_mask_fake = mask_loss(model_mask=d_mask_composite, mask=g_mask_foreground)
        # d_mask_anti_shortcut_loss = mask_loss(model_mask=d_mask_anti_shortcut, mask=g_mask_foreground)
        # d_mask_grounded_fake_loss = mask_loss(model_mask=d_mask_grounded_fake, mask=random_mask)

        # =====================================================
        # WGAN Loss
        d_real = -tf.reduce_mean(score_foreground)
        d_fake = tf.reduce_mean(score_composite)

        g_fake = -tf.reduce_mean(score_composite)
        g_anti_shortcut = tf.reduce_mean(score_anti_shortcut)

        d_grounded_fake = tf.reduce_mean(score_grounded_fake)

        d_mask_real = tf.minimum(tf.reduce_mean(tf.squared_difference(d_mask_foreground, 0)),
                                 tf.reduce_mean(tf.squared_difference(d_mask_foreground, 1)))

        d_mask_fake = tf.minimum(tf.reduce_mean(tf.squared_difference(d_mask_composite, g_mask_foreground)),
                                 tf.reduce_mean(tf.squared_difference(d_mask_composite, (1 - g_mask_foreground))))

        d_mask_anti_shortcut_loss = tf.minimum(
            tf.reduce_mean(tf.squared_difference(d_mask_anti_shortcut, g_mask_foreground)),
            tf.reduce_mean(tf.squared_difference(d_mask_anti_shortcut, (1 - g_mask_foreground))))

        d_mask_grounded_fake_loss = \
            tf.minimum(tf.reduce_mean(tf.squared_difference(d_mask_grounded_fake, random_mask)),
                       tf.reduce_mean(tf.squared_difference(d_mask_grounded_fake, (1 - random_mask))))

        # Aux loss
        L_aux = d_mask_real + d_mask_fake + d_mask_anti_shortcut_loss + d_mask_grounded_fake_loss
        L_g = g_fake + g_anti_shortcut
        L_d = d_real + d_fake + d_grounded_fake + 0.1 * L_aux

        # L_AUX_LOSS_1 = -tf.reduce_mean(tf.squared_difference(g_fake + g_anti_shortcut, 0))
        # L_AUX_LOSS_2 = -tf.reduce_mean(tf.squared_difference(g_fake - d_real, 0))

        g_optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
        d_optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
        # aux_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

        vars = tf.all_variables()
        for var in vars:
            print(var)

        g_vars = [v for v in vars if 'generator' in v.name]
        d_vars = [v for v in vars if 'discriminator' in v.name]

        # ===========================WGAN clip D params===============================
        clip_ops = []
        for var in d_vars:
            clip_bounds = [-.01, .01]
            clip_ops.append(
                tf.assign(
                    var,
                    tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                )
            )
        clip_disc_weights = tf.group(*clip_ops)
        # ============================================================================

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            g_train_op = g_optimizer.minimize(L_g, var_list=g_vars)
            d_train_op = d_optimizer.minimize(L_d, var_list=d_vars)
            # aux_train_op = aux_optimizer.minimize(L_AUX_LOSS_1 + L_AUX_LOSS_2, var_list=g_vars)

        # ________________________________Other Configurations___________________________________________
        init_op = tf.initialize_all_variables()
        saver = tf.train.Saver()

        # _____________________Create a session for running operations in the Graph._____________________
        with tf.Session() as sess:
            # Initialize the variables (like the epoch counter).
            sess.run(init_op)

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for step in range(cfgs.MAX_ITERATION):
                d_epoch_loss = 0
                g_epoch_loss = 0

                image_dict = dict()

                for i in range(max(len(background_images), len(foreground_images))):
                    p_idx = np.random.randint(low=0, high=len(foreground_images))
                    s_idx = np.random.randint(low=0, high=len(background_images))
                    sh_idx = np.random.randint(low=0, high=len(foreground_images))

                    while sh_idx == p_idx:
                        sh_idx = np.random.randint(low=0, high=len(foreground_images))

                    fore = cv2.imread(foreground_images[p_idx]) / 127.5 - 1
                    back = cv2.imread(background_images[s_idx]) / 127.5 - 1
                    sh_fore = cv2.imread(foreground_images[sh_idx]) / 127.5 - 1

                    _, _, d_loss, g_loss, \
                    image_dict['foreground_img'], image_dict['background_img'], \
                    image_dict['shuffled_foreground_img'], image_dict['g_mask_foreground_img'], \
                    image_dict['d_mask_foreground_img'], image_dict['composited_img'], \
                    image_dict['anti_shortcut_img'], image_dict['random_mask_img'], \
                    image_dict['grounded_fake_img'], image_dict['d_mask_composite_img'], \
                    image_dict['d_mask_anti_shortcut_img'], image_dict['d_mask_grounded_fake_img'] = \
                        sess.run([d_train_op,
                                  g_train_op,
                                  L_d,
                                  L_g,
                                  foreground,
                                  background,
                                  shuffled_foreground,
                                  g_mask_foreground,
                                  d_mask_foreground,
                                  composited_image,
                                  anti_shortcut_image,
                                  random_mask,
                                  grounded_fake_image,
                                  d_mask_composite,
                                  d_mask_anti_shortcut,
                                  d_mask_grounded_fake,
                                  ],
                                 feed_dict={
                                     foreground_plh: fore,
                                     background_plh: back,
                                     shuffled_foreground_plh: sh_fore
                                 })
                    d_epoch_loss += d_loss
                    g_epoch_loss += g_loss

                    sess.run(clip_disc_weights)

                logger.info('Epoch {} Generator Loss: {}'.format(step, g_epoch_loss))
                logger.info('Epoch {} Discriminator Loss: {}'.format(step, d_epoch_loss))
                logger.info('\n')

                save_images(image_dict, path=cfgs.RESULT_PATH, epoch=step)

                if step % (cfgs.MAX_ITERATION // 20) == 0:
                    saver.save(sess=sess, save_path=os.path.join(cfgs.CKPT_PATH, 'CPGAN.ckpt'), global_step=step)

            coord.request_stop()
            # Wait for threads to finish.
            coord.join(threads)


def save_images(image_dict, path, epoch):
    '''
    Save images
    :param image_dict: image names and values
    :param path: save path
    :param epoch: epoch
    :return: None
    '''
    logger.info('Saving {} images...'.format(len(image_dict)))

    for img in tqdm(image_dict):
        save_path = os.path.join(path, img + '_' + str(epoch) + '.jpg')
        _save_image(image_dict[img], path=save_path)


def _save_image(image, path):
    '''
    Save one image
    :param image: image value
    :param path: save path
    :return: None
    '''
    image_shape = image.shape
    if len(image_shape) == 4:
        # Save mask
        if 'mask' in path:
            a = 0
            b = 255.
        else:
            a = 1
            b = 127.5
        image = np.reshape((image + a) * b,
                           newshape=[image_shape[-3], image_shape[-2], image_shape[-1]]).astype(np.uint8)
    cv2.imwrite(filename=path, img=image)


def random_process_image(image):
    '''
    Data augmentation
    :param image: image
    :return:
    '''
    image = tf.image.random_crop(image, size=[224, 224, cfgs.CHANNEL])
    image = tf.image.random_flip_left_right(image)

    # Some Image processing policies may not work.
    # image = tf.image.random_brightness(image, max_delta=0.1)
    # image = tf.image.random_contrast(image, lower=0, upper=0.1)
    # image = tf.image.random_hue(image, max_delta=0.1)
    # image = tf.image.random_saturation/(image, lower=0, upper=0.1)
    return image


if __name__ == '__main__':
    foreground_images = glob(os.path.join(cfgs.IMAGE_FOLDER, 'plane', '*.jpg'))
    background_images = glob(os.path.join(cfgs.IMAGE_FOLDER, 'sky', '*.jpg'))
    main()
