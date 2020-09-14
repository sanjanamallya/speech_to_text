    
letter_list = ['@',' ','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',\
            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', '<sos>','<eos>']


def index_to_letter(index_list):
    
    curr_sentence = ""
    for char in index_list:
        if int(char) == 34:
            return curr_sentence 
        else:
            curr_char = idx_to_letter[int(char)]
            curr_sentence += curr_char

    return curr_sentence

def transform_letter_to_index(transcript):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    idx_to_letter  = { i : letter_list[i] for i in range(0, len(letter_list) ) }
    letter_to_idx = {v: k for k, v in idx_to_letter.items()}
    indice_list = list()

    for sentence in transcript:
        words = ''
        for word in sentence:
            curr_word = word.decode('UTF-8')
            words +=   curr_word + ' '
        indice_list_ = [letter_to_idx[c] for c in words]
        del indice_list_[-1]
        indice_list_.insert(0, letter_to_idx['<sos>'])
        indice_list_.append(letter_to_idx['<eos>'])
        indice_list.append(np.asarray(indice_list_))
        # CHECK 
        # <sos>The Boy eats <eos>
        # Check for spaces betwen words and make sure last word and eos doesn't have a space inbetween
        
    indice_array = np.asarray(indice_list)
    return indice_array

