# -*- coding:utf-8 -*-
import sys, random

sys.path.insert(0, "/home/gongxijun/mxnet/python")
import numpy as np
import mxnet as mx

from lstm import LSTM_CTC

from io import BytesIO
from captcha.image import ImageCaptcha
import cv2, random
import os
import argparse

maps = {}
maps_value = 11

for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
    maps[char] = maps_value
    maps_value += 1
image_height = 30
image_width = 100


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None  # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


def get_label(buf):
    global maps
    ret = np.zeros(len(buf))
    for i in range(len(buf)):
        if buf[i].isdigit():
            ret[i] = int(buf[i]) + 1
        else:
            ret[i] = maps[buf[i]]
    return ret


class OCRIter(mx.io.DataIter):
    def __init__(self, batch_size, num_label, init_states, path, check):
        super(OCRIter, self).__init__()
        self.batch_size = batch_size
        self.num_label = num_label
        self.init_states = init_states
        self.path = path
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (batch_size, image_width * image_height))] + init_states
        self.provide_label = [('label', (self.batch_size, num_label))]
        self.check = check

    def __iter__(self):

        init_state_names = [x[0] for x in self.init_states]
        dir_list = os.listdir(self.path)
        pic_num = len(dir_list)
        num = 0
        for k in range(pic_num / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                if num > pic_num - 1:
                    break
                img = cv2.imread(self.path + '/' + dir_list[num], 0)
                lable_value = dir_list[num].split('.')[0]
                lable_value = lable_value.split('_')[1]
                num += 1
                img = cv2.resize(img, (image_width, image_height))
                img = img.transpose(1, 0)  # 转置
                img = img.reshape((image_width * image_height))
                img = np.multiply(img, 1 / 255.0)  # 化整到0~1之间

                data.append(img)
                label.append(get_label(lable_value))

            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass


BATCH_SIZE = 100
SEQ_LENGTH = 30


def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i + 1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret


def Accuracy(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = label[i]
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                    match = False
                    break
            if match:
                hit += 1.0
        total += 1.0
    return hit / total


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='验证码识别参数列表')

    parser.add_argument('--num_hidden', dest='num_hidden',
                        help='lstm神经元个数',
                        default=128, type=int, nargs='?')
    parser.add_argument('--num_lstm_layer', dest='num_lstm_layer',
                        help='lstm层数',
                        default=2, type=int, nargs='?')
    parser.add_argument('--num_epoch', dest='num_epoch',
                        help='迭代的次数',
                        default=160, type=int, nargs='?')
    parser.add_argument('--learning_rate', dest='learning_rate',
                        help='学习速率',
                        default=0.01, type=int, nargs='?')
    parser.add_argument('--momentum', dest='momentum',
                        help='动量,用于梯度算法中',
                        default=0.9, type=int, nargs='?')
    parser.add_argument('--num_label', dest='num_label',
                        help='验证码字符个数',
                        default=4, type=int, nargs='?')
    parser.add_argument('--mode', dest='mode',
                        help='0代表从新开始训练,1代表在原有模型上进行训练',
                        default=0, type=int, nargs='?')
    parser.add_argument('--prefix', dest='prefix',
                        help='模型坐在目录位置model-000x.params,model-symbol.json',
                        default='../model/model', type=str, nargs='?')
    parser.add_argument('--iteration', dest='iteration',
                        help='模型名称迭代次数命名，如model-0080.params写80即可',
                        default=4, type=int, nargs='?')
    parser.add_argument('--gpu', dest='gpu',
                        help='是否使用GPU,如果使用GPU输入使用GPU的数量1,2,3...',
                        default=-1, type=int, nargs='?')

    return parser.parse_args()


if __name__ == '__main__':
    arg = parse_args()

    init_c = [('l%d_init_c' % l, (BATCH_SIZE, arg.num_hidden)) for l in range(arg.num_lstm_layer)]
    init_h = [('l%d_init_h' % l, (BATCH_SIZE, arg.num_hidden)) for l in range(arg.num_lstm_layer)]
    init_states = init_c + init_h

    train_path = '../img_data/train_data'  # '/media/gongxijun/fun/tlbb/train'
    test_path = '../img_data/test_data'  # '/media/gongxijun/fun/tlbb/test'

    data_train = OCRIter(BATCH_SIZE, arg.num_label, init_states, train_path, 'train')
    data_val = OCRIter(BATCH_SIZE, arg.num_label, init_states, test_path, 'test')

    symbol = LSTM_CTC().lstm_unroll(arg.num_lstm_layer, SEQ_LENGTH,
                                    num_hidden=arg.num_hidden,
                                    num_label=arg.num_label)
    contexts = []
    if arg.gpu > 0:
        contexts = [mx.context.gpu(arg.gpu)]
    else:
        contexts = [mx.context.cpu(0)]

        model = mx.model.FeedForward(ctx=contexts,
                                     symbol=symbol,
                                     num_epoch=arg.num_epoch,
                                     learning_rate=arg.learning_rate,
                                     momentum=arg.momentum,
                                     wd=0.00001,
                                     initializer=mx.init.Xavier(factor_type="in",
                                                                magnitude=2.34)) \
            if 0 == arg.mode else \
            mx.model.FeedForward.load(arg.prefix,
                                      arg.iteration,
                                      learning_rate=arg.learning_rate,
                                      ctx=contexts,
                                      numpy_batch_size=BATCH_SIZE,
                                      num_epoch=arg.num_epoch)

import logging

head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head, filename='../train_data.log', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
print 'begin fit'
logger.info('begin fit')
model.fit(X=data_train, eval_data=data_val,
          eval_metric=mx.metric.np(Accuracy),
          batch_end_callback=mx.callback.Speedometer(BATCH_SIZE, 100), logger=logger)
model.save(arg.prefix, arg.iteration + 10)
