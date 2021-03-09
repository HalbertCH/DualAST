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

import argparse
import tensorflow as tf
import os
tf.set_random_seed(228)
from model import Artgan

def parse_list(str_value):
    if ',' in str_value:
        str_value = str_value.split(',')
    else:
        str_value = [str_value]
    return str_value


parser = argparse.ArgumentParser(description='')

# ========================== GENERAL PARAMETERS ========================= #
parser.add_argument('--model_name',
                    dest='model_name',
                    default='model1',
                    help='Name of the model')
parser.add_argument('--phase',
                    dest='phase',
                    default='train',
                    help='Specify current phase: train or inference.')
parser.add_argument('--image_size',
                    dest='image_size',
                    type=int,
                    default=256*3,
                    help='For training phase: will crop out images of this particular size.'
                         'For inference phase: each input image will have the smallest side of this size. '
                         'For inference recommended size is 1280.')


# ========================= TRAINING PARAMETERS ========================= #
parser.add_argument('--ptad',
                    dest='path_to_art_dataset',
                    type=str,
                    #default='./data/vincent-van-gogh_paintings/',
                    default='/disk1/chb/data/vincent-van-gogh_road-with-cypresses-1890',
                    help='Directory with paintings representing style we want to learn.')
parser.add_argument('--ptcd',
                    dest='path_to_content_dataset',
                    type=str,
                    default='/disk1/chb/data/data_large',
                    help='Path to Places365 training dataset.')


parser.add_argument('--total_steps',
                    dest='total_steps',
                    type=int,
                    default=int(3e5),
                    help='Total number of steps')

parser.add_argument('--batch_size',
                    dest='batch_size',
                    type=int,
                    default=1,
                    help='# images in batch')
parser.add_argument('--lr',
                    dest='lr',
                    type=float,
                    default=0.0002,   # 0.0002
                    help='initial learning rate for adam')
parser.add_argument('--save_freq',
                    dest='save_freq',
                    type=int,
                    default=1000,
                    help='Save model every save_freq steps')
parser.add_argument('--ngf',
                    dest='ngf',
                    type=int,
                    default=32,
                    help='Number of filters in first conv layer of generator(encoder-decoder).')
parser.add_argument('--ndf',
                    dest='ndf',
                    type=int,
                    default=64,
                    help='Number of filters in first conv layer of discriminator.')

# Weights of different losses.
parser.add_argument('--dlw',
                    dest='discr_loss_weight',
                    type=float,
                    default=1.,
                    help='Weight of discriminator loss.')
parser.add_argument('--tlw',
                    dest='transformer_loss_weight',
                    type=float,
                    default=100.,
                    help='Weight of transformer loss.')
parser.add_argument('--slw',
                    dest='style_loss_weight',
                    type=float,
                    default=0.3,
                    help='Weight of style loss.')
parser.add_argument('--cflw',
                    dest='content_feature_loss_weight',
                    type=float,
                    default=100.,
                    help='Weight of content feature loss.')
parser.add_argument('--cfadvlw',
                    dest='content_feature_adv_loss_weight',
                    type=float,
                    default=10.,
                    help='Weight of content feature adv loss.')
parser.add_argument('--tvlw',
                    dest='tv_loss_weight',
                    type=float,
                    default=10.,
                    help='Weight of total variation loss.')
parser.add_argument('--dsr',
                    dest='discr_success_rate',
                    type=float,
                    default=0.8,
                    help='Rate of trials that discriminator will win on average.')

# ========================= INFERENCE PARAMETERS ========================= #
parser.add_argument('--ii_dir',
                    dest='inference_images_dir',
                    type=parse_list,
                    default=['./data/sample_photographs/'],
                    help='Directory with content images we want to process.')
parser.add_argument('--reference',
                    dest='reference_image',
                    default='images/reference/van-gogh/1.jpg',
                    help='Directory with content images we want to process.')
parser.add_argument('--save_dir',
                    type=str,
                    default=None,
                    help='Directory to save inference output images.'
                         'If not specified will save in the model directory.')
parser.add_argument('--ckpt_nmbr',
                    dest='ckpt_nmbr',
                    type=int,
                    default=None,
                    help='Checkpoint number we want to use for inference. '
                         'Might be None(unspecified), then the latest available will be used.')

args = parser.parse_args()


def main(_):

    tfconfig = tf.ConfigProto(allow_soft_placement=False)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = Artgan(sess, args)

        if args.phase == 'train':
            model.train(args, ckpt_nmbr=args.ckpt_nmbr)
        if args.phase == 'inference' or args.phase == 'test':
            print("Inference.")
            model.inference(args, args.inference_images_dir, resize_to_original=False,
                            to_save_dir=args.save_dir,
                            ckpt_nmbr=args.ckpt_nmbr)

        if args.phase == 'inference_on_frames' or args.phase == 'test_on_frames':
            print("Inference on frames sequence.")
            model.inference_video(args,
                                  path_to_folder=args.inference_images_dir[0],
                                  resize_to_original=False,
                                  to_save_dir=args.save_dir,
                                  ckpt_nmbr = args.ckpt_nmbr)
        sess.close()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    tf.app.run()
