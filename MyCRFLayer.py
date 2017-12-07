import chainer.links as L
import chainer.functions as F
from chainer import variable

import numpy as np
from chainer.functions.math import sum as _sum
from chainer.functions.math import exponential as _exponential

# More details: https://github.com/createmomo/CRF-Layer-on-the-Top-of-BiLSTM
# This code is modified based on https://github.com/glample/tagger

class My_CRF(L.CRF1d):
    def __init__(self, n_label):
        super(My_CRF, self).__init__(n_label)
        with self.init_scope():
            '''
            [Initialization]
            '''
            # Generate random values for transition matrix.
            # The shape of transition matrix is (n_label+2, n_label+2).
            # "2" means the extra added labels, START and END. (see 3.2)
            drange = np.sqrt(6. / (np.sum((n_label + 2, n_label + 2))))
            value = drange * np.random.uniform(low=-1.0, high=1.0, size=(n_label + 2, n_label + 2))
            transitions = np.array(value, dtype=np.float32)
            self.cost = variable.Parameter(transitions)

            # The number of unique labels in training data set (e.g B-Person, I-Person, O)
            self.n_label = n_label

            # The small value will fill the expanded emission score matrix as described in 3.2
            self.small = -1000

    def __call__(self, xs, ys):
        '''
        :param xs: the outputs of BiLSTM layer (the emission score matrix)
        :param ys: the ground truth labels
        :return: CRF loss
        '''
        '''
        Loss Function
        '''

        # Assign new id for extra added labels (START and END)
        b_id = np.array([self.n_label], dtype='i')
        e_id = np.array([self.n_label + 1], dtype='i')

        total_loss = 0.0
        small = self.small

        #Compute crf loss for each sentence
        for xs_i, ys_i in zip(xs,ys):
            s_len = len(xs_i)# how many words does the sentence have

            # Expand the emission score matrix by adding two extra labels (START and END).
            # For more details, please see the example in 3.2
            b_s = np.array([[small] * self.n_label + [0, small]]).astype(np.float32)
            e_s = np.array([[small] * self.n_label + [small, 0]]).astype(np.float32)
            observations = F.concat((xs_i, small * np.ones((s_len,2),dtype='f')),axis=1)
            observations = F.concat((b_s,observations,e_s), axis=0)

            # Compute the real path score according the ground truth labels (see 2.4)
            # Emission score of the real path
            real_path_score = _sum.sum(xs_i[list(range(s_len)), ys_i])

            # Transition score of the real path
            padded_tags_ids = F.concat((b_id, ys_i, e_id), axis=0)

            real_path_score += _sum.sum(self.cost[
                padded_tags_ids[list(range(s_len + 1))].data,
                padded_tags_ids[[current_item + 1 for current_item in range(s_len + 1)]].data
            ])

            # Compute the score of all the possible paths of current sentence (see 2.5)
            all_paths_scores = self.forward(observations, self.cost)

            # The crf cost of current sentence (see 2.5)
            current_cost = - (real_path_score - all_paths_scores)

            total_loss += current_cost

        return total_loss

    def log_sum_exp(self, x, axis=None):
        '''
        Compute the sum of scores in log space (see 2.5).
        This function is used in forward.
        '''
        xmax = F.max(x, axis=axis, keepdims=True)
        xmax_ = F.max(x, axis=axis)

        second_item = _exponential.log(_sum.sum(_exponential.exp(x - F.broadcast_to(xmax,x.shape)),axis=axis))

        return xmax_ + second_item

    def forward(self, observations, transitions,
                viterbi=False,
                return_best_sequence=False
                ):
        '''
        :param observations: (see 2.5) In 2.5, 'obs' are the observations here.
        :param transitions: Transition score matrix
        :param viterbi: When the viterbi and return_best_sequence are true, this method will return the predicted best paths.
        If false, this function will return the sum of scores in log space
        :param return_best_sequence: Please see above.
        :return: The sum of scores in log space or the predicted best sequence
        '''
        '''
        This function is described in detail in 2.5 and 2.6.
        '''
        def recurrence(obs, previous, transitions):
            previous = previous.reshape((previous.shape[0],1))
            obs = obs.reshape((1,obs.shape[0]))
            if viterbi:# Please see 2.6
                scores = F.broadcast_to(previous,(self.n_label+2, self.n_label+2)) + F.broadcast_to(obs,(self.n_label+2, self.n_label+2)) + transitions
                scores = scores.data
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)
                    out2 = np.array(out2,dtype='i')
                    return out, out2
            else:# Please see 2.5 (Return the sum of scores in log space)
                previous = F.broadcast_to(previous,(self.n_label+2, self.n_label+2))
                obs = F.broadcast_to(obs,(self.n_label+2, self.n_label+2))
                return self.log_sum_exp(previous + obs + transitions, axis=0)

        def mini_function_for_best_sequence(beta_i, previous):
            return beta_i[previous]

        if return_best_sequence:# Return the best predicted path for one sentence (see 2.6)
            initial_0 = observations[0]
            alpha_0 = np.array(initial_0.data, dtype='f')
            alpha_0 = F.expand_dims(alpha_0,axis=0)

            alpha_1 = None

            flag = True
            for obs in observations[1:]:
                initial_0, initial_1 = recurrence(obs, initial_0, transitions)
                alpha_0 = F.vstack((alpha_0,F.expand_dims(initial_0,axis=0)))

                if flag == True:
                    alpha_1 = np.array(initial_1, dtype='i')
                    alpha_1 = F.expand_dims(alpha_1, axis=0)
                    flag = False

                alpha_1 = F.vstack((alpha_1, F.expand_dims(initial_1, axis=0)))

            alpha_0 = alpha_0.data[1:]

            initial_beta = np.argmax(alpha_0[-1])
            initial_beta = initial_beta.astype('i')
            sequence = np.array(initial_beta,dtype='i')
            sequence = F.expand_dims(sequence,axis=0)

            for item in alpha_1.data[::-1].astype('i'):
                initial_beta = mini_function_for_best_sequence(item,initial_beta)
                sequence = F.concat((sequence, F.expand_dims(np.array(initial_beta), axis=0)), axis=0)

            sequence = sequence[::-1][2:-1]
            sequence = sequence.reshape(1,sequence.shape[0])
            return sequence[0]# Return best path

        else:# Please see 2.5 (Return the sum of scores in log space)
            initial = observations[0]
            alpha = []
            alpha.append(initial)

            for obs in observations[1:]:
                initial = recurrence(obs, initial, transitions)
                alpha.append(initial)

            alpha = alpha[1:]

            return self.log_sum_exp(alpha[-1], axis=0)

    def argmax(self, xs):
        '''
        :param xs: The list of new sentences
        :return: Predicted labels for the new sentences
        '''
        best_sequence = []

        small = self.small

        # Predict the labels for new sentences (Please see 2.6)
        for xs_i in xs:
            s_len = len(xs_i)

            b_s = np.array([[small] * self.n_label + [0, small]]).astype(np.float32)
            e_s = np.array([[small] * self.n_label + [small, 0]]).astype(np.float32)
            observations = F.concat((xs_i, small * np.ones((s_len, 2), dtype='f')), axis=1)
            observations = F.concat((b_s, observations, e_s), axis=0)

            current_best_sequence = self.forward(observations, self.cost, viterbi=True, return_best_sequence=True)
            best_sequence.append(current_best_sequence.data)

        return best_sequence

