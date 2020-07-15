import os
import math
import time
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from tqdm import trange
from datetime import datetime

from loss import LossHelper
from network import SVBRDF_Net
from tqdm import tqdm
import numpy as np
from render import Render

from dataset2 import Dataset2


def train(args):
    now = datetime.now().strftime("%y%m%d-%H%M%S")

    tag = args.tag
    if len(tag) > 0:
        tag = '-' + args.tag
    run_dir = os.path.join(os.path.dirname(__file__), '../output', now + tag)

    for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
    strategy = tf.distribute.OneDeviceStrategy(device='/gpu:{}'.format(0))

    # train_dataset = [Dataset2('C:/Users/Administrator/Desktop/new_data/pbr/pbr/', include_src=False, dstype=0), Dataset2('D:/Bin/images/', include_src=True, dstype=1)]

    train_dataset = Dataset2('C:/Users/Administrator/Desktop/new_data/pbr/pbr/', include_src=False, dstype=0)
    helper = LossHelper()
    # tensorboard writer
    os.makedirs(run_dir, exist_ok=True)

    model = SVBRDF_Net(512, 24)
    # model = tf.keras.models.load_model('../output/200618-105008/model_0260.h5')
    # model.summary()

    # optimizer
    lr_schedule = tf.optimizers.schedules.ExponentialDecay(
        args.lr, decay_steps=10000, decay_rate=0.97, staircase=False)

    # new_lr = lr_schedule(260)
    # lr_schedule = tf.optimizers.schedules.ExponentialDecay(
    #     new_lr, decay_steps=10000, decay_rate=0.97, staircase=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) #, clipnorm=3.0
    render = Render()   


    # Run at each step
    @tf.function
    def model_step(sample, training=True):
        svbrdf = {
            'diffuse': sample[0],
            'specular': sample[1],
            'glossiness': sample[3],
            'normal': sample[2]
        }
        images = tf.reshape(sample[4], list(sample[4].shape)[:-2] + [-1])
        # rendered_images = render.render(svbrdf, unflatten=False)

        pred = model(tf.expand_dims(images, 0), training=training)
        pred_svbrdf = {
            'diffuse': pred[0],
            'specular': pred[1],
            'glossiness': pred[2],
            'normal': pred[3]
        }
        pred_images = render.render(pred_svbrdf, unflatten=False)
        
        loss0 = helper.svbrdf(pred_svbrdf, svbrdf)
        # loss1 = helper.images(pred_images, rendered_images)
        total_loss = loss0 #* 0.1 + loss1

        if training:
            gradients = optimizer.get_gradients(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return svbrdf, images, pred_svbrdf, pred_images


    # Run at each epoch
    def run_epoch(iterator, epoch, step, prefix, training=True):
        helper.reset()
        # with strategy.scope():
        t_step = tqdm(range(step), ascii=True, leave=False, desc='Training' if training else 'Evaluating')
        try:
            for s in t_step:
                # # print("Step: ", s)
                # if s % 2 == 0:
                #     # item = iterator[0].nxt() #next(iterator)
                #     item = iterator[1].nxt()
                # else:
                #     item = iterator[1].nxt()

                item = iterator.nxt()
                # print(item)
                sample, images, pred_svbrdf, pred_images = model_step(item, training=training)
                # print(lr_schedule(s))
                #refined_svbrdf, refined_images = refine(pred_svbrdf, images)
                
            # return the loss
        except KeyboardInterrupt:
            t_step.close()
            raise
        return helper.get_svbrdf_loss().numpy() #, helper.get_images_loss().numpy()

    try:
        for epoch in range(args.epoch+1):
            print('\nEpoch: {}/{}'.format(epoch, args.epoch))
            train_loss = run_epoch(train_dataset, epoch, args.step, 'training', True)
            # eval_loss = run_epoch(eval_iter,  epoch, args.eval_step, 'evaluate', False)

            tl = 'Train loss: SVBRDF={:.4f}'.format(train_loss)
            el = '' # 'Eval loss: SVBRDF={:.4f}, image={:.4f}'.format(eval_loss[0], eval_loss[1])
            print('{}\t{}'.format(tl, el))

            if epoch % args.save == 0:
                save_path = os.path.join(run_dir, 'model_{:04d}.h5'.format(epoch))
                model.save(save_path)

    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=600, help="number of epoches")
    parser.add_argument('-step', type=int, default=1000, help="number of steps per epoch")
    parser.add_argument('-eval_step', type=int, default=100, help="number of evaluate steps per epoch")
    parser.add_argument('-save', type=int, default=5, help="save model per number of epoches")
    parser.add_argument('-lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('-batch', type=int, default=4, help="batch size for training")
    parser.add_argument('-loglevel', type=int, default=40, help="logging level")
    parser.add_argument('-tag', default='', help="tag")
    parser.add_argument('-gpu', type=int, default=0, help="run on which GPU")
    parser.add_argument('-model', default=None, help='train from model')
    args = parser.parse_args()

    train(args)
