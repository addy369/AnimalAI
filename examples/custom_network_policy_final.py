import warnings
from itertools import zip_longest
from abc import ABC, abstractmethod
from stable_baselines.common.policies import RecurrentActorCriticPolicy, register_policy, nature_cnn

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

import gym

from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc,ortho_init,_ln#, lstm
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution
from stable_baselines.common.input import observation_input

def residual_block_fixup_attention(inputs, num_layers, name):
    depth = inputs.get_shape()[-1].value
    stddev = np.sqrt(2.0/(inputs.get_shape()[1].value * inputs.get_shape()[2].value * inputs.get_shape()[3].value * num_layers))

    bias0 = tf.get_variable(name + "/bias0", [], initializer=tf.zeros_initializer())
    bias1 = tf.get_variable(name + "/bias1", [], initializer=tf.zeros_initializer())
    bias2 = tf.get_variable(name + "/bias2", [], initializer=tf.zeros_initializer())
    bias3 = tf.get_variable(name + "/bias3", [], initializer=tf.zeros_initializer())

    multiplier = tf.get_variable(name + "/multiplier", [], initializer=tf.ones_initializer())
    out = tf.nn.elu(inputs)
    out = out + bias0
    out = tf.layers.conv2d(out, depth, 3, padding='same', name='res1/' + name, use_bias = False, kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
    out = out + bias1
    out = out * channel_attention(out, depth)
    out = tf.nn.elu(out)
    out = out + bias2
    out = tf.layers.conv2d(out, depth, 3, padding='same', name='res2/' + name, use_bias = False, kernel_initializer=tf.zeros_initializer())
    out = out * multiplier + bias3

    return out + inputs

def animal_conv_residual_fixup_attention(inputs, depths = [16,32,32,64]):
    out = inputs
    i = 0
    for depth in depths:
        i = i + 1
        out = tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + str(i))
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block_fixup_attention(out, i, name='rb1' + str(i))
        out = residual_block_fixup_attention(out, i, name='rb2' + str(i))
        print('shapes of layer_' + str(i), str(out.get_shape().as_list()))
    out = tf.nn.elu(out)
    return out

def lstm(extracted_features, dones_ph, cell_state_hidden, scope, n_hidden, n_env,n_steps,init_scale=1.0,layer_norm=False):
    """
    Creates an Long Short Term Memory (LSTM) cell for TensorFlow

    :param extracted_features: (TensorFlow Tensor) The input tensor for the LSTM cell (before converting into sequence)
    :param dones_ph: (TensorFlow Tensor) The mask tensor for the LSTM cell (before converting into sequence)
    :param cell_state_hidden: (TensorFlow Tensor) The state tensor for the LSTM cell
    :param scope: (str) The TensorFlow variable scope
    :param n_hidden: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :param layer_norm: (bool) Whether to apply Layer Normalization or not
    :return: (TensorFlow Tensor) LSTM cell
    """
    #_, n_input = [v.value for v in input_tensor[0].get_shape()]
    n_input = extracted_features.get_shape()[1].value
    print(n_input)
    print(extracted_features.get_shape())
    input_sequence = batch_to_seq(extracted_features, n_env, n_steps)
    masks = batch_to_seq(dones_ph, n_env, n_steps)
    print(len(input_sequence))
    print(len(masks))
    with tf.variable_scope(scope):
        weight_x = tf.get_variable("wx", [n_input, n_hidden * 4], initializer=ortho_init(init_scale))
        weight_h = tf.get_variable("wh", [n_hidden, n_hidden * 4], initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", [n_hidden * 4], initializer=tf.constant_initializer(0.0))

        if layer_norm:
            # Gain and bias of layer norm
            gain_x = tf.get_variable("gx", [n_hidden * 4], initializer=tf.constant_initializer(1.0))
            bias_x = tf.get_variable("bx", [n_hidden * 4], initializer=tf.constant_initializer(0.0))

            gain_h = tf.get_variable("gh", [n_hidden * 4], initializer=tf.constant_initializer(1.0))
            bias_h = tf.get_variable("bh", [n_hidden * 4], initializer=tf.constant_initializer(0.0))

            gain_c = tf.get_variable("gc", [n_hidden], initializer=tf.constant_initializer(1.0))
            bias_c = tf.get_variable("bc", [n_hidden], initializer=tf.constant_initializer(0.0))

    cell_state, hidden = tf.split(axis=1, num_or_size_splits=2, value=cell_state_hidden)
    for idx, (_input, mask) in enumerate(zip(input_sequence, masks)):
        cell_state = cell_state * (1 - mask)
        hidden = hidden * (1 - mask)
        if layer_norm:
            gates = _ln(tf.matmul(_input, weight_x), gain_x, bias_x) \
                    + _ln(tf.matmul(hidden, weight_h), gain_h, bias_h) + bias
        else:
            print(_input.get_shape())
            print(weight_x.get_shape())
            print(hidden.get_shape())
            print(weight_h.get_shape())
            gates = tf.matmul(_input, weight_x) + tf.matmul(hidden, weight_h) + bias
        in_gate, forget_gate, out_gate, cell_candidate = tf.split(axis=1, num_or_size_splits=4, value=gates)
        in_gate = tf.nn.sigmoid(in_gate)
        forget_gate = tf.nn.sigmoid(forget_gate)
        out_gate = tf.nn.sigmoid(out_gate)
        cell_candidate = tf.tanh(cell_candidate)
        cell_state = forget_gate * cell_state + in_gate * cell_candidate
        if layer_norm:
            hidden = out_gate * tf.tanh(_ln(cell_state, gain_c, bias_c))
        else:
            hidden = out_gate * tf.tanh(cell_state)
        input_sequence[idx] = hidden
    cell_state_hidden = tf.concat(axis=1, values=[cell_state, hidden])
    return input_sequence, cell_state_hidden

class LstmPolicy(RecurrentActorCriticPolicy):
    """
    Policy object that implements actor critic, using LSTMs.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture. Notation similar to the
        format described in mlp_extractor but with additional support for a 'lstm' entry in the shared network part.
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None,
                 net_arch=['lstm',dict(vf=[1],pi=[9])], act_fun=tf.tanh, cnn_extractor=nature_cnn, layer_norm=False, feature_extraction="cnn", 
                 **kwargs):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm, ), reuse=reuse,
                                         scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if net_arch is None:  # Legacy mode
            if layers is None:
                layers = [64, 64]
            else:
                warnings.warn("The layers parameter is deprecated. Use the net_arch parameter instead.")

            with tf.variable_scope("model", reuse=reuse):
                
                if feature_extraction == "cnn":
                    active = tf.nn.relu
                    conv_1=self.processed_obs
            
                    conv_1 = tf.layers.Conv2D(
                             filters=8,
                             kernel_size=[5,5],
                             strides=(4, 4),
                             activation=tf.nn.elu).apply(conv_1)    

                    for i, layer_size in enumerate([16, 32 ,64]):

                        conv_1 = active(tf.layers.Conv2D(filters=layer_size, kernel_size=(3,3),strides=(2,2),name='conv' + str(i)).apply(conv_1))
            ## Need to add velocity and time inputs??
                    conv1 = animal_conv_residual_fixup_attention(conv1, depths = [16,32,64,128])
                    extracted_features = tf.layers.flatten(conv_1)
                    extracted_features = active(tf.layers.dense(extracted_features, 1024, name='fc'))
                    print("hello")
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    for i, layer_size in enumerate(layers):
                        extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                            init_scale=np.sqrt(2)))
                print(extracted_features)
                #input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)

                #masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                print(input_sequence)
                rnn_output, self.snew = lstm(extracted_features, self.dones_ph, self.states_ph, 'lstm1', n_lstm,self.n_env,self.n_steps,
                                             layer_norm=layer_norm)
                print("hello")
                rnn_output = seq_to_batch(rnn_output)
                value_fn = linear(rnn_output, 'vf', 1)

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

            self._value_fn = value_fn
        else:  # Use the new net_arch parameter (new_arch=[512,dict(vf=[1], pi=[9])])
            if layers is not None:
                warnings.warn("The new net_arch parameter overrides the deprecated layers parameter.")
            # if feature_extraction == "cnn":
            #     raise NotImplementedError()
            with tf.variable_scope("model", reuse=reuse):
                if feature_extraction == "cnn":
                    active = tf.nn.relu
                    conv_1=self.processed_obs
            
                    conv_1 = tf.layers.Conv2D(
                             filters=8,
                             kernel_size=[5,5],
                             strides=(4, 4),
                             activation=tf.nn.elu).apply(conv_1)    

                    for i, layer_size in enumerate([16, 32 ,64]):

                        conv_1 = active(tf.layers.Conv2D(filters=layer_size, kernel_size=(3,3),strides=(2,2),name='conv' + str(i)).apply(conv_1))
            ## Need to add velocity and time inputs??

                    extracted_features = tf.layers.flatten(conv_1)
                    extracted_features = active(tf.layers.dense(extracted_features, 1024, name='fc'))
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    for i, layer_size in enumerate(layers):
                        extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                            init_scale=np.sqrt(2)))

                latent = tf.layers.flatten(self.processed_obs)
                policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
                value_only_layers = []  # Layer sizes of the network that only belongs to the value network

                # Iterate through the shared layers and build the shared parts of the network
                lstm_layer_constructed = False
                for idx, layer in enumerate(net_arch):
                    if isinstance(layer, int):  # Check that this is a shared layer
                        layer_size = layer
                        latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
                        #raise ValueError("The net_arch parameter must only contain one occurrence of 'lstm'!")
                    elif layer == "lstm":
                        if lstm_layer_constructed:
                            raise ValueError("The net_arch parameter must only contain one occurrence of 'lstm'!")
                        #input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
                        #masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                        rnn_output, self.snew = lstm(extracted_features, self.dones_ph, self.states_ph, 'lstm1', n_lstm,self.n_env,self.n_steps,
                                                     layer_norm=layer_norm)
                        latent = seq_to_batch(rnn_output)
                        lstm_layer_constructed = True
                    else:
                        assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                        if 'pi' in layer:
                            assert isinstance(layer['pi'],
                                              list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                            policy_only_layers = layer['pi']

                        if 'vf' in layer:
                            assert isinstance(layer['vf'],
                                              list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                            value_only_layers = layer['vf']
                        break  # From here on the network splits up in policy and value network

                # Build the non-shared part of the policy-network
                latent_policy = latent
                for idx, pi_layer_size in enumerate(policy_only_layers):
                    if pi_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the policy network.")
                    assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                    latent_policy = act_fun(
                        linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

                # Build the non-shared part of the value-network
                latent_value = latent
                for idx, vf_layer_size in enumerate(value_only_layers):
                    if vf_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the value function "
                                                  "network.")
                    assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                    latent_value = act_fun(
                        linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

                if not lstm_layer_constructed:
                    raise ValueError("The net_arch parameter must contain at least one occurrence of 'lstm'!")

                self._value_fn = linear(latent_value, 'vf', 1)
                # TODO: why not init_scale = 0.001 here like in the feedforward
                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(latent_policy, latent_value)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})


    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})


    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
