# --
# hmm model

import numpy as np
from hmmlearn import hmm


# lambda function
softmax = lambda x : np.exp(x) / np.sum(np.exp(x))


class HMMvModel:
  """
  simple HMM model
  """

  def __init__(self, n_states, n_symbols, connect_type='sequence'):

    # arguments
    self.n_states = n_states
    self.n_symbols = n_symbols
    self.connect_type = connect_type

    # transition probability matrix (random init)
    #self.A = np.stack([softmax(np.random.uniform(low=0.0, high=1.0, size=(n_states))) for n in range(n_states)], axis=0)
    self.A = np.random.uniform(low=0.0, high=1.0, size=(n_states, n_states))

    # sequence connection
    if connect_type == 'sequence':

      # masking for audio sequence
      self.A = np.triu(self.A)

      # softmax for tiangular matrix
      self.A = [np.pad(softmax(self.A[n, :][self.A[n, :] != 0]), (n, 0)) for n in range(n_states)]

    # interconnected
    else:

      # softmax
      self.A = [softmax(self.A[n, :]) for n in range(n_states)]

    # stack
    self.A = np.stack(self.A, axis=0)

    # emmision prob matrix
    self.B = np.random.uniform(low=0.0, high=1.0, size=(n_states, n_symbols))
    self.B = np.stack([softmax(self.B[n, :]) for n in range(n_states)], axis=0)

    # init prob
    self.Pi = softmax(np.random.uniform(low=0.0, high=1.0, size=n_states))

    # state dict
    self.state_dict = { 'init_prob': self.Pi, 'trans_p_matrix': self.A, 'emmision_p_matrix': self.B }

    print(self.A)


  def print_states(self):
    """
    print state dict
    """
    [print("\n{}:\n {}".format(k,v)) for k, v in self.state_dict.items()]


if __name__ == '__main__':
  """
  main
  """

  # states
  #S = { 1: 'cat',  2: 'cow',  3: 'pig' }
  S = { 1: 'still',  2: 'walking',  3: 'run' }

  # speed
  V = ['speed']
  
  # symbols
  #V = { 'size': np.range(0, 10),  'color': np.range(0, 3) }

  # observation sequence (class correspondence)
  O = np.array([1, 2, 3, 2, 2, 3, 2, 2, 1, 0])

  # states corresponding to observations
  S_o = np.array([0, 2, 5, 3, 2, 6, 3, 1, 0, 0])

  #[print(V[o]) for o in O]

  # sength
  T = len(O)
  n_symbols = len(V)
  n_states = 1

  # create hmm model
  hmm_model1 = HMMvModel(n_states=n_states, n_symbols=n_symbols)
  hmm_model2 = hmm.CategoricalHMM(n_components=3)

  hmm_model1.print_states()

  hmm_model2.startprob = hmm_model1.Pi.copy()
  hmm_model2.transmat = hmm_model1.A.copy()
  hmm_model2.emmissionprob_ = hmm_model1.B.copy()

  print(O)
  print(O.reshape(-1, 1))

  # input
  x = np.array([1, 2, 3, 1, 2, 3, 3, 3, 3])

  hmm_model2.fit(O.reshape(-1, 1))

  # state sequence
  y = hmm_model2.predict(x.reshape(-1, 1))
  print("output m2: ", y)




