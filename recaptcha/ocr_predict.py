# coding=utf-8
# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
import os

sys.path.insert(0, "/home/gongxijun/mxnet/python")
import numpy as np
import mxnet as mx

from lstm_model import LSTMInferenceModel

import cv2, random
import threading

BATCH_SIZE = 100
SEQ_LENGTH = 30
image_height = 30
image_width = 100
num_hidden = 128
num_lstm_layer = 2

num_epoch = 10
momentum = 0.9
num_label = 4
iteration = 40
n_channel = 1
contexts = [mx.context.cpu(0)]
_, arg_params, __ = mx.model.load_checkpoint('../model/model', iteration)

maps = {}
maps_value = 1

for char in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
    maps[maps_value] = char
    maps_value += 1

root_path = "../img_data/test_data"


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


if __name__ == '__main__':

    model = LSTMInferenceModel(num_lstm_layer,
                               SEQ_LENGTH,
                               num_hidden=num_hidden,
                               num_label=num_label,
                               arg_params=arg_params,
                               data_size=n_channel *image_width*image_height,
                               ctx=contexts[0])
    cnt = 0
    for image_path in os.listdir(root_path):

        img = cv2.imread(os.path.join(root_path, image_path), 0)
        img = cv2.resize(img, (image_width, image_height))
        img = img.transpose(1, 0)
        img = img.reshape((1, image_width*image_height))
        img = np.multiply(img, 1 / 255.0)
        prob = model.forward(mx.nd.array(img), new_seq=True)

        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(prob[k]))
        p = ctc_label(p)

        # print 'Predicted label: ' + str(p)

        pred = ''
        for c in p:
            pred += maps[int(c)]
        real_str= image_path.split('.')[0].split('_')[-1]
        if pred.__eq__(real_str):
            cnt+=1
        print  cnt
        print 'Predicted number: ' + pred + "  实际值： " + real_str
