# Copyright (C) 2018  Artsiom Sanakoyeu and Dmytro Kotovenko
#
# This file is part of Adaptive Style Transfer
#
# Adaptive Style Transfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Adaptive Style Transfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import division
from ops import *
from vgg19 import Vgg19


def encoder(image, options, reuse=True, name="encoder"):
    """
    Args:
        image: input tensor, must have
        options: options defining number of kernels in conv layers
        reuse: to create new encoder or use existing
        name: name of the encoder

    Returns: Encoded image.
    """

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        image = instance_norm(input=image,
                              is_training=options.is_training,
                              name='g_e0_bn')
        c0 = tf.pad(image, [[0, 0], [15, 15], [15, 15], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(input=conv2d(c0, options.gf_dim, 3, 1, padding='VALID', name='g_e1_c'),
                                      is_training=options.is_training,
                                      name='g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(input=conv2d(c1, options.gf_dim, 3, 2, padding='VALID', name='g_e2_c'),
                                      is_training=options.is_training,
                                      name='g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim * 2, 3, 2, padding='VALID', name='g_e3_c'),
                                      is_training=options.is_training,
                                      name='g_e3_bn'))
        c4 = tf.nn.relu(instance_norm(conv2d(c3, options.gf_dim * 4, 3, 2, padding='VALID', name='g_e4_c'),
                                      is_training=options.is_training,
                                      name='g_e4_bn'))
        c5 = tf.nn.relu(instance_norm(conv2d(c4, options.gf_dim * 8, 3, 2, padding='VALID', name='g_e5_c'),
                                      is_training=options.is_training,
                                      name='g_e5_bn'))
        return c5


def SCB(image, gf_dim, name='style_extractor'):
    with tf.variable_scope(name):
        vgg_style = Vgg19(image)
        style_layers = [vgg_style.relu2_1, vgg_style.relu3_1, vgg_style.relu4_1, vgg_style.relu5_1]

        sf = []
        k = [2, 4, 8, 8]
        for i in range(len(style_layers)):
            x = tf.pad(style_layers[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            x = conv2d(x, gf_dim * k[i], 4, 2, padding='VALID', name='conv1_' + str(i))
            x = tf.nn.relu(x)

            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            x = conv2d(x, gf_dim * k[i], 4, 2, padding='VALID', name='conv2_' + str(i))
            x = tf.nn.relu(x)

            x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)  # global average pooling
            x = conv2d(x, gf_dim * k[i], 1, 1, padding='VALID', name='SE_logit'+str(i))
            # x = z_score(x)
            sf.append(x)

        return sf


def decoder(features, style, options, n_res=9, reuse=True, name="decoder"):
    """
    Args:
        features: input tensor, must have
        options: options defining number of kernels in conv layers
        reuse: to create new decoder or use existing
        name: name of the encoder

    Returns: Decoded image.
    """

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        x = features
        num_kernels = features.get_shape().as_list()[-1]
        # Now stack 9 residual blocks
        for i in range(n_res):
            x = residule_block(x, num_kernels, name='g_r' + str(i))

        # Style-control block.
        sf = SCB(style, options.gf_dim)
        x = features * sf[3]

        # Decode image.
        d1 = deconv2d(x, options.gf_dim * 8, 3, 2, name='g_d1_dc')    # channel: 256
        d1 = tf.nn.relu(instance_norm(input=d1,
                                      name='g_d1_bn',
                                      is_training=options.is_training))
        d1 *= sf[2]

        d2 = deconv2d(d1, options.gf_dim * 4, 3, 2, name='g_d2_dc')    # channel: 128
        d2 = tf.nn.relu(instance_norm(input=d2,
                                      name='g_d2_bn',
                                      is_training=options.is_training))
        # d2 *= sf[1]

        d3 = deconv2d(d2, options.gf_dim * 2, 3, 2, name='g_d3_dc')    # channel: 64
        d3 = tf.nn.relu(instance_norm(input=d3,
                                      name='g_d3_bn',
                                      is_training=options.is_training))
        # d3 *= sf[0]

        d4 = deconv2d(d3, options.gf_dim, 3, 2, name='g_d4_dc')         # channel: 32
        d4 = tf.nn.relu(instance_norm(input=d4,
                                      name='g_d4_bn',
                                      is_training=options.is_training))

        d4 = tf.pad(d4, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.sigmoid(conv2d(d4, 3, 7, 1, padding='VALID', name='g_pred_c'))*2. - 1.
        return pred


def discriminator(image, options, reuse=True, name="discriminator"):
    """
    Discriminator agent, that provides us with information about image plausibility at
    different scales.
    Args:
        image: input tensor
        options: options defining number of kernels in conv layers
        reuse: to create new discriminator or use existing
        name: name of the discriminator

    Returns:
        Image estimates at different scales.
    """
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(instance_norm(conv2d(image, options.df_dim * 2, ks=5, name='d_h0_conv'),
                   name='d_bn0'))
        h0_pred = conv2d(h0, 1, ks=5, s=1, name='d_h0_pred', activation_fn=None)

        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim * 2, ks=5, name='d_h1_conv'),
                                 name='d_bn1'))
        h1_pred = conv2d(h1, 1, ks=10, s=1, name='d_h1_pred', activation_fn=None)

        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim * 4, ks=5, name='d_h2_conv'),
                                 name='d_bn2'))

        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim * 8, ks=5, name='d_h3_conv'),
                                 name='d_bn3'))
        h3_pred = conv2d(h3, 1, ks=10, s=1, name='d_h3_pred', activation_fn=None)

        h4 = lrelu(instance_norm(conv2d(h3, options.df_dim * 8, ks=5, name='d_h4_conv'),
                                 name='d_bn4'))

        h5 = lrelu(instance_norm(conv2d(h4, options.df_dim * 16, ks=5, name='d_h5_conv'),
                                 name='d_bn5'))
        h5_pred = conv2d(h5, 1, ks=6, s=1, name='d_h5_pred', activation_fn=None)

        h6 = lrelu(instance_norm(conv2d(h5, options.df_dim * 16, ks=5, name='d_h6_conv'),
                                 name='d_bn6'))
        h6_pred = conv2d(h6, 1, ks=3, s=1, name='d_h6_pred', activation_fn=None)

        return {"scale_0": h0_pred,
                "scale_1": h1_pred,
                "scale_3": h3_pred,
                "scale_5": h5_pred,
                "scale_6": h6_pred}


def feature_discriminator(feature, options, reuse=True, name="feature_dis"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(instance_norm(conv2d(feature, options.df_dim * 4, ks=5, name='d_h0_conv'), name='d_bn0'))
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim * 8, ks=5, name='d_h1_conv'), name='d_bn1'))
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim * 16, ks=5, name='d_h2_conv'), name='d_bn2'))

        logits = tf.reduce_sum(h2, [1, 2], keep_dims=False)
        logits = fully_connected(logits, 1, scope='fc')

        return logits


# ====== Define different types of losses applied to discriminator's output. ====== #

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_-target))

def mse_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def z_score(x):
    mean, variance = tf.nn.moments(x, axes=3)
    return (x - mean) / variance

def style_statistic_loss(style_image, noise_image, epsilon=1e-5):
    vgg_style = Vgg19(style_image)
    vgg_train = Vgg19(noise_image)

    train_layers = [vgg_train.relu1_1, vgg_train.relu2_1, vgg_train.relu3_1, vgg_train.relu4_1, vgg_train.relu5_1]
    style_layers = [vgg_style.relu1_1, vgg_style.relu2_1, vgg_style.relu3_1, vgg_style.relu4_1, vgg_style.relu5_1]

    loss = 0
    for i in range(len(train_layers)):
        train_mean, train_var = tf.nn.moments(train_layers[i], axes=[1, 2], keep_dims=True)
        train_std = tf.sqrt(train_var + epsilon)

        style_mean, style_var = tf.nn.moments(style_layers[i], axes=[1, 2], keep_dims=True)
        style_std = tf.sqrt(style_var + epsilon)

        mean_loss = tf.reduce_sum(tf.squared_difference(train_mean, style_mean))
        std_loss = tf.reduce_sum(tf.squared_difference(train_std, style_std))
        loss += mean_loss + std_loss

    loss = loss / tf.to_float(len(train_layers))
    return loss


def gram(inputs):
    shape = tf.shape(inputs)
    B = shape[0]
    H = shape[1]
    W = shape[2]
    C = shape[3]

    HW = H * W
    HWC = HW * C
    feats = tf.reshape(inputs, (B, HW, C))
    feats_T = tf.transpose(feats, perm=[0, 2, 1])

    g = tf.matmul(feats_T, feats) / tf.to_float(HWC)

    return g

def style_gram_loss(style_image, noise_image):
    vgg_style = Vgg19(style_image)
    vgg_train = Vgg19(noise_image)

    train_layers = [vgg_train.relu1_1, vgg_train.relu2_1, vgg_train.relu3_1, vgg_train.relu4_1, vgg_train.relu5_1]
    style_layers = [vgg_style.relu1_1, vgg_style.relu2_1, vgg_style.relu3_1, vgg_style.relu4_1, vgg_style.relu5_1]

    loss = 0
    for i in range(len(train_layers)):
        G = gram(train_layers[i])
        A = gram(style_layers[i])

        size = tf.size(train_layers[i])
        loss += tf.nn.l2_loss(G - A) * 2 / tf.to_float(size)

    loss = loss / tf.to_float(len(train_layers))
    return loss


def total_variation_loss(img):
    shape = tf.shape(img)
    B = shape[0]
    H = shape[1]
    W = shape[2]
    C = shape[3]

    tv_y_size = (H - 1) * W * C
    tv_x_size = H * (W - 1) * C
    y_tv = tf.nn.l2_loss(img[:, 1:, :, :] - img[:, :(H - 1), :, :])
    x_tv = tf.nn.l2_loss(img[:, :, 1:, :] - img[:, :, :(W - 1), :])

    loss = 2 * (y_tv / tf.to_float(tv_y_size) + x_tv / tf.to_float(tv_x_size)) / tf.to_float(B)

    return loss

def adv_feature_loss(real_outputs, fake_outputs):
    d_loss = -tf.reduce_mean(tf.minimum(0., -1.0 + real_outputs)) - tf.reduce_mean(tf.minimum(0., -1.0 - fake_outputs))
    g_loss = - tf.reduce_mean(fake_outputs)
    return d_loss, g_loss


def reduce_spatial_dim(input_tensor):
    """
    Since labels and discriminator outputs are of different shapes (and even ranks)
    we should write a routine to deal with that.
    Args:
        input: tensor of shape [batch_size, spatial_resol_1, spatial_resol_2, depth]
    Returns:
        tensor of shape [batch_size, depth]
    """
    input_tensor = tf.reduce_mean(input_tensor=input_tensor, axis=1)
    input_tensor = tf.reduce_mean(input_tensor=input_tensor, axis=1)
    return input_tensor


def add_spatial_dim(input_tensor, dims_list, resol_list):
    """
        Appends dimensions mentioned in dims_list resol_list times. S
        Args:
            input: tensor of shape [batch_size, depth0]
            dims_list: list of integers with position of new  dimensions to append.
            resol_list: list of integers with corresponding new dimensionalities for each dimension.
        Returns:
            tensor of new shape
        """
    for dim, res in zip(dims_list, resol_list):

        input_tensor = tf.expand_dims(input=input_tensor,  axis=dim)
        input_tensor = tf.concat(values=[input_tensor]*res, axis=dim)
    return input_tensor


def repeat_scalar(input_tensor, shape):
    """
    Repeat scalar values.
    :param input_tensor: tensor of shape [batch_size, 1]
    :param shape: new_shape of the element of the tensor
    :return: tensor of the shape [batch_size, *shape] with elements repeated.
    """
    with tf.control_dependencies([tf.assert_equal(tf.shape(input_tensor)[1], 1)]):
        batch_size = tf.shape(input_tensor)[0]
    input_tensor = tf.tile(input_tensor, tf.stack(values=[1, tf.reduce_prod(shape)], axis=0))
    input_tensor = tf.reshape(input_tensor, tf.concat(values=[[batch_size], shape, [1]], axis=0))
    return input_tensor


def transformer_block(input_tensor, kernel_size=10):
    """
    This is a simplified version of transformer block described in our paper
    https://arxiv.org/abs/1807.10201.
    Args:
        input_tensor: Image(or tensor of rank 4) we want to transform.
        kernel_size: Size of kernel we apply to the input_tensor.
    Returns:
        Transformed tensor
    """
    return slim.avg_pool2d(inputs=input_tensor, kernel_size=kernel_size, stride=1, padding='SAME')
