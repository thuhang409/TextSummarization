from keras import backend as K 
from .attention import AttentionLayer
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from keras.models import load_model

class model:
  def __init__(self):
    self.model = None
    
  def LSTM_seq2seq(self,latent_dim, emb_dim, vocab_size, encoder_input_len, decoder_input_len, encoder_emb_weight, decoder_emb_weight):
    K.clear_session()

    # ENCODER
    encoder_inputs = Input(shape=(None,))

    # encoder embedding layer 
    encoder_embedding_layer = Embedding(input_dim = vocab_size + 1, 
                                        output_dim = emb_dim,
                                        input_length = encoder_input_len,
                                        weights = [encoder_emb_weight],
                                        trainable = False)
    enc_emb = encoder_embedding_layer(encoder_inputs)

    # encoder lstm 1
    encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

    # encoder lstm 2
    encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    # encoder lstm 3
    encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

    # DECODER
    decoder_inputs = Input(shape=(None,))

    # decoder embedding layer 
    decoder_embedding_layer = Embedding(input_dim = vocab_size + 1, 
                                        output_dim = emb_dim,
                                        input_length = decoder_input_len,
                                        weights = [decoder_emb_weight],
                                        trainable = False)
    dec_emb = decoder_embedding_layer(decoder_inputs)

    # decoder lstm
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
    decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

    # ATTENTION layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

    # CONCAT layer?
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    # Dense layer
    decoder_dense = TimeDistributed(Dense(vocab_size+1, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)
    
    # MODEL
    self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    self.model.summary()

    return self.model, encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, decoder_lstm, decoder_embedding_layer, attn_layer, decoder_dense

  def Train(self, x_train, y_train, x_test, y_test, path_model, epochs, batch_size):
    self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    es = [EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2), 
          ModelCheckpoint(path_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)]  
    
    self.model.save_weights(path_model.format(epoch=0))

    history = self.model.fit([x_train,y_train[:,:-1]], \
                  y_train.reshape(x_train.shape[0],y_train.shape[1], 1)[:,1:], \
                  epochs=epochs,callbacks=[es],batch_size=batch_size, \
                  validation_data= ([x_test,y_test[:,:-1]], y_test.reshape(y_test.shape[0],y_test.shape[1], 1)[:,1:]))
    
    return history           
  
  def Saved_model(self,model,path_model):
    model.save(path_model)
    print("Saved model.")

  def Load_model(self, path_model):
    model = load_model(path_model)
    print("Loaded model from disk")
    return model

  @staticmethod
  def LoadEncoderModel(encoder_inputs, encoder_outputs, state_h, state_c):
    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])
    return encoder_model

  @staticmethod
  def LoadDecoderModel(decoder_embedding_layer, decoder_inputs, decoder_lstm, attn_layer, decoder_dense, latent_dim, MAX_LEN_ART):
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(MAX_LEN_ART,latent_dim))
    
    # Get the embeddings of the decoder sequence
    dec_emb2= decoder_embedding_layer(decoder_inputs) 
    
    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])
    
    #attention inference
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])
    
    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_outputs2 = decoder_dense(decoder_inf_concat) 
    
    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])
    return decoder_model
