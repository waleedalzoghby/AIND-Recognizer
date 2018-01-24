import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    
    for word_id in range(0, len(test_set.get_all_Xlengths())):
        current_word_feature_lists_sequences, current_sequences_length = test_set.get_item_Xlengths(word_id)
        word_log_likelihoods = {}

        for word, model in models.items():
            try:
                scr = model.score(current_word_feature_lists_sequences, current_sequences_length)
                word_log_likelihoods[word] = scr
            except:
                word_log_likelihoods[word] = float("-inf")
                continue
        probabilities.append(word_log_likelihoods)

        guesses.append(max(word_log_likelihoods, key = word_log_likelihoods.get))

    return probabilities, guesses
