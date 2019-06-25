import sys

from zoo.models.common import KerasZooModel
from zoo.models.recommendation import Recommender
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import *

if sys.version >= '3':
    long = int
    unicode = str


class SessionRecommender(Recommender):

    def __init__(self, item_count, item_embed, mlp_hidden_layers, rnn_hidden_layers,
                 include_history, seq_length, his_length, bigdl_type="float"):
        self.item_count = int(item_count)
        self.item_embed = int(item_embed)
        self.mlp_hidden_layers = [int(unit) for unit in mlp_hidden_layers]
        self.rnn_hidden_layers = [int(unit) for unit in rnn_hidden_layers]
        self.include_history = include_history
        self.seq_length = int(seq_length)
        self.his_length = int(his_length)
        self.bigdl_type = bigdl_type
        self.model = self.build_model()
        super(SessionRecommender, self).__init__(None, self.item_count, self.item_embed,
                                                 self.mlp_hidden_layers, self.rnn_hidden_layers,
                                                 self.include_history, self.seq_length,
                                                 self.his_length, self.bigdl_type,
                                                 self.model)

    def build_model(self):
        input_rnn = Input(shape=(self.seq_length, ))
        session_table = Embedding(self.item_count + 1, self.item_count, init="uniform")(input_rnn)

        gru = GRU(self.rnn_hidden_layers[0], return_sequences=True)(session_table)
        for hidden in range(1, len(self.rnn_hidden_layers) - 1):
            gru = GRU(self.rnn_hidden_layers[hidden], return_sequences=True)(gru)
        gru_last = GRU(self.rnn_hidden_layers[-1], return_sequences=False)(gru)
        rnn = Dense(self.item_count)(gru_last)

        if self.include_history:
            input_mlp = Input(shape=(self.his_length,))
            his_table = Embedding(self.item_count + 1, self.item_count, init="uniform")(input_mlp)
            flatten = Flatten()(his_table)
            mlp = Dense(self.mlp_hidden_layers[0], activation="relu")(flatten)
            for hidden in range(1, len(self.mlp_hidden_layers)):
                mlp = Dense(self.mlp_hidden_layers[hidden], activation="relu")(mlp)
            mlp_last = Dense(self.item_count)(mlp)
            merged = merge(inputs=[mlp_last, rnn], mode="sum")
            out = Activation(activation="softmax")(merged)
            model = Model(input=[input_mlp, input_rnn], output=out)
        else:
            out = Activation(activation="softmax")(rnn)
            model = Model(input=input_rnn, output=out)
        return model

    @staticmethod
    def load_model(path, weight_path=None, bigdl_type="float"):
        """
        Load an existing SessionRecommender model (with weights).

        # Arguments
        path: The path for the pre-defined model.
              Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        weight_path: The path for pre-trained weights if any. Default is None.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadSessionRecommender", path, weight_path)
        model = KerasZooModel._do_load(jmodel, bigdl_type)
        model.__class__ = SessionRecommender
        return model





