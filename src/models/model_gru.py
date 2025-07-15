from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense, Flatten, Layer
from keras.optimizers import Adam
from keras.metrics import MeanAbsolutePercentageError, RootMeanSquaredError
from keras import backend as K

def build_gru(seq_length, n_features, lookahead):
    model = Sequential()
    model.add(GRU(32, return_sequences=True, input_shape=(seq_length, n_features)))
    model.add(Dropout(0.3))
    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(128))
    model.add(Dropout(0.2))
    model.add(Dense(lookahead, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse', metrics=[MeanAbsolutePercentageError(), RootMeanSquaredError()])
    return model

class AttentionLayer(Layer):
    def __init__(self,**kwargs):
        super(AttentionLayer,self).__init__(**kwargs)
    def build(self,input_shape):
        self.w=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal", trainable=True)
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros", trainable=True)
        super(AttentionLayer,self).build(input_shape)
    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.w)+self.b), axis=-1)
        at=K.softmax(et, axis=-1)
        at=K.expand_dims(at)
        output=x*at
        return K.sum(output,axis=1)
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])
    def get_config(self):
        return super(AttentionLayer,self).get_config()

def build_gru_attention(seq_length, n_features, lookahead):
    model = Sequential()
    model.add(GRU(32, return_sequences=True, input_shape=(seq_length, n_features)))
    model.add(Dropout(0.3))
    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(AttentionLayer())
    model.add(Flatten())
    model.add(Dense(lookahead, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse', metrics=[MeanAbsolutePercentageError(), RootMeanSquaredError()])
    return model
