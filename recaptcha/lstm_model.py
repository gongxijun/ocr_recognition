# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys

sys.path.insert(0, "/home/gongxijun/mxnet/python")
import numpy as np
import mxnet as mx

from lstm import LSTMState, LSTMParam, LSTM_CTC


class LSTMInferenceModel(object):
    def __init__(self,
                 num_lstm_layer,
                 seq_len,
                 num_hidden,
                 num_label,
                 arg_params,
                 data_size, ctx=mx.cpu()):
        self.num_lstm_layer = num_lstm_layer
        self.seq_len = seq_len
        self.num_hidden = num_hidden
        self.num_label = num_label
        self.arg_params = arg_params
        self.data_size = data_size
        self.ctx = ctx
        lstm_ctc = LSTM_CTC()
        self.sym = lstm_ctc.lstm_inference_symbol(self.num_lstm_layer,
                                                  self.seq_len,
                                                  self.num_hidden,
                                                  self.num_label)

        batch_size = 1
        init_c = [('l%d_init_c' % l, (batch_size, self.num_hidden)) for l in range(self.num_lstm_layer)]
        init_h = [('l%d_init_h' % l, (batch_size, self.num_hidden)) for l in range(self.num_lstm_layer)]
        data_shape = [("data", (batch_size, self.data_size))]
        input_shapes = dict(init_c + init_h + data_shape)
        self.executor = self.sym.simple_bind(ctx=self.ctx, **input_shapes)

        for key in self.executor.arg_dict.keys():
            if key in self.arg_params:
                self.arg_params[key].copyto(self.executor.arg_dict[key])

        self.state_name = []
        for i in range(self.num_lstm_layer):
            self.state_name.append("l%d_init_c" % i)
            self.state_name.append("l%d_init_h" % i)
        self.states_dict = dict(zip(self.state_name, self.executor.outputs[1:]))
        self.input_arr = mx.nd.zeros(data_shape[0][1])

    def forward(self, input_data, new_seq=False):

        # type: (object, object, object, object, object, object, object) -> object

        if new_seq:
            for key in self.states_dict.keys():
                self.executor.arg_dict[key][:] = 0.
        input_data.copyto(self.executor.arg_dict["data"])
        self.executor.forward()
        for key in self.states_dict.keys():
            self.states_dict[key].copyto(self.executor.arg_dict[key])
        prob = self.executor.outputs[0].asnumpy()
        return prob
