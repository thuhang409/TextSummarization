from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np

class predict:
  def __init__(self):
    pass

  def decode_sequence(self, encoder_model, decoder_model, input_seq, token, MAX_LEN_SUM, sostok = 'sostok', eostok = 'eostok'):
    reverse_target_word_index = token.index_word
    target_word_index = token.word_index
  
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index[sostok]

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!=eostok):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == eostok  or len(decoded_sentence.split()) >= (MAX_LEN_SUM-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

  @staticmethod
  def seq2text(seq, tokenizer):
    return ' '.join([tokenizer.index_word[i] for i in seq if i != 0])


  
