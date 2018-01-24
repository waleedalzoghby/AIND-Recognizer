import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        #warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def calc_p(self, num_states, features):
        return ( num_states ** 2 ) + ( 2 * num_states * features ) - 1

    def calc_score_bic(self, log_likelihood, p, num_data_points):
        return (-2 * log_likelihood) + (p * np.log(num_data_points))

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        #warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on BIC scores
        bic_scores = []
        for states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(states)
                log_likelihood = hmm_model.score(self.X, self.lengths)
                data_points = sum(self.lengths)
                N, f = self.X.shape
                p = self.calc_p(states, f)
                score_bic = self.calc_score_bic(log_likelihood, p, data_points)
                bic_scores.append(tuple([score_bic, hmm_model]))
            except:
                pass
        return min(bic_scores, key = lambda x: x[0])[1] if bic_scores else None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def calc_log_likelihood_remaining_words(self, model, remaining_words):
        return [model[1].score(word[0], word[1]) for word in remaining_words]
    
    def select(self):
        #warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on DIC scores
        remaining_words = []
        
        dic_scores = []
        models = []
        for word in self.words:
            if word != self.this_word:
                remaining_words.append(self.hwords[word])
        try:
            for states in range(self.min_n_components, self.max_n_components + 1):
                hmm_model = self.base_model(states)
                log_likelihood_original_word = hmm_model.score(self.X, self.lengths)
                models.append((log_likelihood_original_word, hmm_model))

        except Exception as e:
            pass
        
        for index, model in enumerate(models):
            log_likelihood_original_word, hmm_model = model
            score_dic = log_likelihood_original_word - np.mean(self.calc_log_likelihood_remaining_words(model, remaining_words))
            dic_scores.append(tuple([score_dic, model[1]]))
        return max(dic_scores, key = lambda x: x[0])[1] if dic_scores else None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        #warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection using CV
        
        kf = KFold(n_splits = 3, shuffle = False, random_state = None)
        likelihoods = []
        cv_scores = []
        
        for states in range(self.min_n_components, self.max_n_components + 1):    
            try:
                if len(self.sequences) > 2:

                    for train_index, test_index in kf.split(self.sequences):
 
                        self.X, self.lengths = combine_sequences(train_index, self.sequences)

                        X_test, lengths_test = combine_sequences(test_index, self.sequences)

                        hmm_model = self.base_model(states)
                        log_likelihood = hmm_model.score(X_test, lengths_test)
                else:
                    hmm_model = self.base_model(states)
                    log_likelihood = hmm_model.score(self.X, self.lengths)

                    likelihoods.append(log_likelihood)
                
                score_cvs_avg = np.mean(likelihoods)
                cv_scores.append(tuple([score_cvs_avg, hmm_model]))
                
            except Exception as e:
                pass

        return max(cv_scores, key = lambda x: x[0])[1] if cv_scores else None