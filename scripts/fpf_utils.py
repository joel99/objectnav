"""Classes copied directly from Matt Golub's fixed-point-finder and
recurrent-whisperer and adapted for Python 3.7
"""

'''
AdaptiveGradNormClip.py
Written using Python 2.7.12
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''
import os
import numpy as np
import pickle

class AdaptiveGradNormClip(object):
	"""Class for managing adaptive gradient norm clipping for stabilizing any gradient-descent-like procedure.
	Essentially, just a running buffer of gradient norms from the last n gradient steps, with a hook into the x-th percentile of those values, which is intended to be used to set the ceiling on the gradient applied at the next iteration of a gradient-descent-like procedure.
	The standard usage is as follows:
	```python
	# Set hyperparameters as desired.
	agnc_hps = dict()
	agnc_hps['sliding_window_len'] = 1.0
	agnc_hps['percentile'] = 95
	agnc_hps['init_clip_val' = 1.0
	agnc_hps['verbose'] = False
	agnc = AdaptiveGradNormClip(**agnc_hps)
	while some_conditions(...):
		# This loop defines one step of the training procedure.
		gradients = get_gradients(data, params)
		grad_norm = compute_gradient_norm(gradients)
		clip_val = agnc.update(grad_norm)
		clipped_gradients = clip_gradients(gradients, clip_val)
		params = apply_gradients(clipped_gradients)
		# (Optional): Occasionally save model checkpoints along with the
		# AdaptiveGradNormClip object (for seamless restoration of a training
		# session)
		if some_other_conditions(...):
			save_checkpoint(params, ...)
			agnc.save(...)
	```
	"""

	''' Included for ready access by RecurrentWhisperer
		(before initializing an instance) '''
	default_hps = {
		'do_adaptive_clipping': True,
		'sliding_window_len': 128,
		'percentile': 95.0,
		'init_clip_val': 1e12,
		'max_clip_val': 1e12,
		'verbose': False
		}

	def __init__(self,
		do_adaptive_clipping=default_hps['do_adaptive_clipping'],
		sliding_window_len=default_hps['sliding_window_len'],
		percentile=default_hps['percentile'],
		init_clip_val=default_hps['init_clip_val'],
		max_clip_val=default_hps['max_clip_val'],
		verbose=default_hps['verbose']):
		'''Builds an AdaptiveGradNormClip object
		Args:
			A set of optional keyword arguments for overriding the default
			values of the following hyperparameters:
			do_adaptive_clipping: A bool indicating whether to implement adaptive gradient norm clipping (i.e., the purpose of this class). Setting to False leads to clipping at a fixed gradient norm specified by fixed_clip_val. Default: True
			sliding_window_len: An int specifying the number of recent steps to
			record. Default: 100.
			percentile: A float between 0.0 and 100.0 specifying the percentile
			of the recorded gradient norms at which to set the clip value.
			Default: 95.
			init_clip_val: A float specifying the initial clip value (i.e., for
			step 1, before any empirical gradient norms have been recorded).
			Default: 1e12.
				This default effectively prevents any clipping on iteration one.
				This has the unfortunate side effect of throwing the vertical
				axis scale on the corresponding Tensorboard plot. The
				alternatives are computationally inefficient: either clip at an
				arbitrary level (or at 0) for the first epoch or compute a
				gradient at step 0 and initialize to the norm of the global
				gradient.
			max_clip_val: A positive float indicating the largest allowable  clipping value. This effectively overrides the adaptive nature of the gradient clipping once the adaptive clip value exceeds this threshold. When do_adaptive_clipping is set to False, this clipping value is always applied at each step. Default: 1e12.
			verbose: A bool indicating whether or not to print status updates.
			Default: False.
		'''
		self.step = 0
		self.do_adaptive_clipping = do_adaptive_clipping
		self.sliding_window_len = sliding_window_len
		self.percentile = percentile
		self.max_clip_val = max_clip_val
		self.grad_norm_log = []
		self.verbose = verbose
		self.save_filename = 'norm_clip.pkl'

		if self.do_adaptive_clipping:
			self.clip_val = init_clip_val
		else:
			self.clip_val = self.max_clip_val

	def __call__(self):
		'''Returns the current clip value.
		Args:
			None.
		Returns:
			A float specifying the current clip value.
		'''
		return self.clip_val

	def update(self, grad_norm):
		'''Update the log of recent gradient norms and the corresponding
		recommended clip value.
		Args:
			grad_norm: A float specifying the gradient norm from the most
			recent gradient step.
		Returns:
			None.
		'''
		if self.do_adaptive_clipping:
			if self.step < self.sliding_window_len:
				# First fill up an entire "window" of values
				self.grad_norm_log.append(grad_norm)
			else:
				# Once the window is full, overwrite the oldest value
				idx = np.mod(self.step, self.sliding_window_len)
				self.grad_norm_log[idx] = grad_norm

			proposed_clip_val = \
				np.percentile(self.grad_norm_log, self.percentile)

			self.clip_val = min(proposed_clip_val, self.max_clip_val)

		self.step += 1

	def save(self, save_dir):
		'''Saves the current AdaptiveGradNormClip state, enabling seamless restoration of gradient descent training procedure.
		Args:
			save_dir: A string containing the directory in which to save the
			current object state.
		Returns:
			None.
		'''

		if self.verbose:
			print('Saving AdaptiveGradNormClip.')
		save_path = os.path.join(save_dir, self.save_filename)
		file = open(save_path,'w')
		file.write(pickle.dumps(self.__dict__))
		file.close

	def restore(self, restore_dir):
		'''Loads a previously saved AdaptiveGradNormClip state, enabling seamless restoration of gradient descent training procedure.
		Args:
			restore_dir: A string containing the directory from which to load
			a previously saved object state.
		Returns:
			None.
		'''
		if self.verbose:
			print('Restoring AdaptiveGradNormClip.')
		restore_path = os.path.join(restore_dir, self.save_filename)
		file = open(restore_path,'r')
		restore_data = file.read()
		file.close()
		self.__dict__ = pickle.loads(restore_data)



'''
AdaptiveLearningRate.py
Written using Python 2.7.12
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''
import os
import pdb
import pickle
import numpy as np
import numpy.random as npr

# if os.environ.get('DISPLAY','') == '':
# 	# Ensures smooth running across environments, including servers without
# 	# graphical backends.
# 	print('No display found. Using non-interactive Agg backend.')
# 	import matplotlib
# 	matplotlib.use('Agg')
import matplotlib.pyplot as plt

class AdaptiveLearningRate(object):
	'''Class for managing an adaptive learning rate schedule based on the
	recent history of loss values. The adaptive schedule begins with an
	optional warm-up period, during which the learning rate logarithmically
	increases up to the initial rate. For the remainder of the training
	procedure, the learning rate will increase following a period of monotonic
	improvements in the loss and will decrease if a loss is encountered that
	is worse than all losses in the recent period. Hyperparameters control the
	length of each of these periods and the extent of each type of learning
	rate change.
	Note that this control flow is asymmetric--stricter criteria must be met
	for increases than for decreases in the learning rate This choice 1)
	encourages decreases in the learning rate when moving into regimes with a
	flat loss surface, and 2) attempts to avoid instabilities that can arise
	when the learning rate is too high (and the often irreversible
	pathological parameter updates that can result). Practically,
	hyperparameters may need to be tuned to optimize the learning schedule and
	to ensure that the learning rate does not explode.
	See test(...) to simulate learning rate trajectories based on specified
	hyperparameters.
	The standard usage is as follows:
	# Set hyperparameters as desired.
	alr_hps = dict()
	alr_hps['initial_rate'] = 1.0
	alr_hps['min_rate'] = 1e-3
	alr_hps['max_n_steps'] = 1e4
	alr_hps['n_warmup_steps'] = 0
	alr_hps['warmup_scale'] = 1e-3
	alr_hps['warmup_shape'] = 'gaussian'
	alr_hps['do_decrease_rate'] = True
	alr_hps['min_steps_per_decrease'] = 5
	alr_hps['decrease_factor'] = 0.95
	alr_hps['do_increase_rate'] = True
	alr_hps['min_steps_per_increase'] = 5
	alr_hps['increase_factor'] = 1./0.95
	alr_hps['verbose'] = False
	alr = AdaptiveLearningRate(**alr_hps)
	# This loop iterates through the optimization procedure.
	while ~alr.is_finished():
		# Get the current learning rate
		learning_rate = alr()
		# Use the current learning rate to update the model parameters.
		# Get the loss of the model after the update.
		params, loss = run_one_training_step(params, learning_rate, ...)
		# Update the learning rate based on the most recent loss value
		# and an internally managed history of loss values.
		alr.update(loss)
		# (Optional): Occasionally save model checkpoints along with the
		# AdaptiveLearningRate object (for seamless restoration of a training
		# session)
		if some_other_conditions(...):
			save_checkpoint(params, ...)
			alr.save(...)
	'''

	''' Included for ready access by RecurrentWhisperer
		(before initializing an instance) '''
	default_hps = {
		'initial_rate': 1.0,
		'min_rate': 1e-3,
		'max_n_steps': 1e4,
		'n_warmup_steps': 0,
		'warmup_scale': 1e-3,
		'warmup_shape': 'gaussian',
		'do_decrease_rate': True,
		'min_steps_per_decrease': 5,
		'decrease_factor': 0.95,
		'do_increase_rate': True,
		'min_steps_per_increase': 5,
		'increase_factor': 1/0.95,
		'verbose': False
		}

	def __init__(self,
		initial_rate = default_hps['initial_rate'],
		min_rate = default_hps['min_rate'],
		max_n_steps = default_hps['max_n_steps'],
		n_warmup_steps = default_hps['n_warmup_steps'],
		warmup_scale = default_hps['warmup_scale'],
		warmup_shape = default_hps['warmup_shape'],
		do_decrease_rate = default_hps['do_decrease_rate'],
		min_steps_per_decrease = default_hps['min_steps_per_decrease'],
		decrease_factor = default_hps['decrease_factor'],
		do_increase_rate = default_hps['do_increase_rate'],
		min_steps_per_increase = default_hps['min_steps_per_increase'],
		increase_factor = default_hps['increase_factor'],
		verbose = default_hps['verbose']):
		'''Builds an AdaptiveLearningRate object
		Args:
			A set of optional keyword arguments for overriding the default
			values of the following hyperparameters:
			initial_rate: Non-negative float specifying the initial learning
			rate. Default: 1.0.
			min_rate: Non-negative float specifying the largest learning
			rate for which is_finished() returns False. This can optionally be
			used externally to signal termination of the optimization
			procedure. This argument is never used internally--the learning
			rate behavior doesn't depend on this value. Default: 1e-3.
			max_n_steps: Non-negative integer specifying the maximum number of
			steps before is_finished() will return True. This can optionally be
			used externally to signal termination of the optimization
			procedure. This argument is never used internally--the learning
			rate behavior doesn't depend on this value. Default: 1e4.
			n_warmup_steps: Non-negative int specifying the number of warm-up
			steps to take. During these warm-up steps, the learning rate will
			monotonically increase up to initial_rate (according to
			warmup_scale and warmup_shape). Default: 0 (i.e., no
			warm-up).
			warmup_scale: Float between 0 and 1 specifying the learning rate
			on the first warm-up step, relative to initial_rate. The first
			warm-up learning rate is warmup_scale * initial_rate. Default:
			0.001.
			warmup_shape: String indicating the shape of the increasing
			learning rate during the warm-up period. Options are 'exp'
			(exponentially increasing learning rates; slope increases
			throughout) or 'gaussian' (slope increases, then decreases; rate
			ramps up faster and levels off smoother than with 'exp').
			Default: 'gaussian'.
			do_decrease_rate: Bool indicating whether or not to decrease the
			learning rate during training (after any warm-up). Default: True.
			min_steps_per_decrease: Non-negative int specifying the number
			of recent steps' loss values to consider when deciding whether to
			decrease the learning rate. Learning rate decreases are made when
			a loss value is encountered that is worse than every loss value in
			this window. When the learning rate is decreased, no further
			decreases are considered until this many new steps have
			transpired. Larger values will slow convergence due to the
			learning rate. Default 5.
			decrease_factor: Float between 0 and 1 specifying the extent of
			learning rate decreases. Whenever a decrease is made, the learning
			rate decreases from x to decrease_factor * x. Values closer to 1
			will slow convergence due to the learning rate. Default: 0.95.
			do_increase_rate: Bool indicating whether or not to increase the
			learning rate during training (after any warm-up). Default: True.
			min_steps_per_increase: Non-negative int specifying the number
			of recent steps' loss values to consider when deciding whether to
			increase the learning rate. Learning rate increases are made when
			the loss has monotonically decreased over this many steps. When
			the learning rate is increased, no further increases are
			considered until this many new steps have transpired. Default 5.
			increase_factor: Float greater than 1 specifying the extent of
			learning rate increases. Whenever an increase is made, the
			learning rate increases from x to increase_factor * x. Larger
			values will slow convergence due to the learning rate. Default:
			1./0.95.
			verbose: Bool indicating whether or not to print status updates.
			Default: False.
		'''

		self.step = 0
		self.step_last_update = -1
		self.prev_rate = None
		self.loss_log = []

		self.initial_rate = initial_rate
		self.min_rate = min_rate
		self.max_n_steps = max_n_steps
		self.do_decrease_rate = do_decrease_rate
		self.decrease_factor = decrease_factor
		self.min_steps_per_decrease = min_steps_per_decrease
		self.do_increase_rate = do_increase_rate
		self.increase_factor = increase_factor
		self.min_steps_per_increase = min_steps_per_increase

		self.n_warmup_steps = n_warmup_steps
		self.warmup_scale = warmup_scale
		self.warmup_shape = warmup_shape

		self.save_filename = 'learning_rate.pkl'

		self._validate_hyperparameters()

		self.warmup_rates = self._get_warmup_rates()

		self.verbose = verbose

		if n_warmup_steps > 0:
			self.learning_rate = self.warmup_rates[0]
		else:
			self.learning_rate = initial_rate

		if self.verbose:
			print('AdaptiveLearningRate schedule requires at least %s steps:' %
				str(self.min_steps))

	def __call__(self):
		'''Returns the current learning rate.'''

		return self.learning_rate

	def is_finished(self, do_check_step=True, do_check_rate=True):
		''' Indicates termination of the optimization procedure. Note: this
		function is never used internally and does not influence the behavior
		of the adaptive learning rate.
		Args:
			do_check_step: Bool indicating whether to check if the step has
			reached max_n_steps.
			do_check_rate: Bool indicating whether to check if the learning rate
			has fallen below min_rate.
		Returns:
			Bool indicating whether any of the termination criteria have been
			met.
		'''

		if do_check_step and self.step > self.max_n_steps:
			return True
		elif self.step <= self.n_warmup_steps:
			return False
		elif do_check_rate and self.learning_rate <= self.min_rate:
			return True
		else:
			return False

	@property
	def min_steps(self):
		''' Computes the minimum number of steps required before the learning
		rate falls below the min_rate, i.e., assuming the rate decreases at
		every opportunity permitted by the properties of this
		AdaptiveLearningRate object.
		Args:
			None.
		Returns:
			An int specifying the minimum number of steps in the adaptive
			learning rate schedule.
		'''
		n_decreases = np.ceil(np.divide(
			(np.log(self.min_rate) - np.log(self.initial_rate)),
			np.log(self.decrease_factor)))
		return self.n_warmup_steps + self.min_steps_per_decrease * n_decreases

	def update(self, loss):
		'''Updates the learning rate based on the most recent loss value
		relative to the recent history of loss values.
		Args:
			loss: A float indicating the loss from the current training step.
		Returns:
			A float indicating the updated learning rate.
		'''
		self.loss_log.append(loss)

		step = self.step
		cur_rate = self.learning_rate
		step_last_update = self.step_last_update

		self.prev_rate = cur_rate

		if step <= self.n_warmup_steps:
			'''If step indicates that we are still in the warm-up, the new rate is determined entirely based on the warm-up schedule.'''
			if step < self.n_warmup_steps:
				self.learning_rate = self.warmup_rates[step]
				if self.verbose:
					print('Warm-up (%d of %d): Learning rate set to %.2e'
						  % (step+1,self.n_warmup_steps,self.learning_rate))
			else: # step == n_warmup_steps:
				self.learning_rate = self.initial_rate
				if self.verbose:
					print('Warm-up complete (or no warm-up). Learning rate set to %.2e'
						  % self.learning_rate)
			self.step_last_update = step

			'''Otherwise, rate may be kept, increased, or decreased based on
			recent loss history.'''
		elif self._conditional_decrease_rate():
			self.step_last_update = step
		elif self._conditional_increase_rate():
			self.step_last_update = step

		self.step = step + 1

		return self.learning_rate

	def save(self, save_dir):
		'''Saves the current state of the AdaptiveLearningRate object.
		Args:
			save_dir: A string containing the directory in which to save.
		Returns:
			None.
		'''
		if self.verbose:
			print('Saving AdaptiveLearningRate.')
		save_path = os.path.join(save_dir, self.save_filename)
		file = open(save_path,'w')
		file.write(pickle.dumps(self.__dict__))
		file.close

	def restore(self, restore_dir):
		'''Restores the state of a previously saved AdaptiveLearningRate
		object.
		Args:
			restore_dir: A string containing the directory in which to find a
			previously saved AdaptiveLearningRate object.
		Returns:
			None.
		'''
		if self.verbose:
			print('Restoring AdaptiveLearningRate.')
		restore_path = os.path.join(restore_dir, self.save_filename)
		file = open(restore_path,'r')
		restore_data = file.read()
		file.close()
		self.__dict__ = pickle.loads(restore_data)

	def _validate_hyperparameters(self):
		'''Checks that critical hyperparameters have valid values.
		Args:
			None.
		Returns:
			None.
		Raises:
			Various ValueErrors depending on the violating hyperparameter(s).
		'''
		def assert_non_negative(attr_str):
			'''
			Args:
				attr_str: The name of a class variable.
			Returns:
				None.
			Raises:
				ValueError('%s must be non-negative but was %d' % (...))
			'''
			val = getattr(self, attr_str)
			if val < 0:
				raise ValueError('%s must be non-negative but was %d'
								 % (attr_str, val))

		assert_non_negative('initial_rate')
		assert_non_negative('n_warmup_steps')
		assert_non_negative('min_steps_per_decrease')
		assert_non_negative('min_steps_per_increase')

		if self.decrease_factor > 1.0 or self.decrease_factor < 0.:
			raise ValueError('decrease_factor must be between 0 and 1, '
			                 'but was %f' % self.decrease_factor)

		if self.increase_factor < 1.0:
			raise ValueError('increase_factor must be >= 1, but was %f'
							 % self.increase_factor)

		if self.warmup_shape not in ['exp', 'gaussian']:
			raise ValueError('warmup_shape must be \'exp\' or \'gaussian\', '
			                 'but was %s' % self.warmup_shape)

	def _get_warmup_rates(self):
		'''Determines the warm-up schedule of learning rates, culminating at
		the desired initial rate.
		Args:
			None.
		Returns:
			An [n_warmup_steps,] numpy array containing the learning rates for
			each step of the warm-up period.
		'''
		scale = self.warmup_scale
		warmup_start = scale*self.initial_rate
		warmup_stop = self.initial_rate

		if self.warmup_shape == 'exp':
			n = self.n_warmup_steps + 1
			warmup_rates = np.logspace(
				np.log10(warmup_start), np.log10(warmup_stop), n)
		elif self.warmup_shape == 'gaussian':
			mu = np.float32(self.n_warmup_steps)
			x = np.arange(mu)

			# solve for sigma s.t. warmup_rates[0] = warmup_start
			sigma = np.sqrt(-mu**2.0 / (2.0*np.log(warmup_start/warmup_stop)))

			warmup_rates = warmup_stop*np.exp((-(x-mu)**2.0)/(2.0*sigma**2.0))

		return warmup_rates

	def _conditional_increase_rate(self):
		'''Increases the learning rate if loss values have monotonically
		decreased over the past n steps, and if no learning rate changes have
		been made in the last n steps, where n=min_steps_per_increase.
		Args:
			None.
		Returns:
			A bool indicating whether the learning rate was increased.
		'''

		did_increase_rate = False
		n = self.min_steps_per_increase

		if self.do_increase_rate and self.step>=(self.step_last_update + n):

			batch_loss_window = self.loss_log[-(1+n):]
			lastBatchLoss = batch_loss_window[-1]

			if all(np.less(batch_loss_window[1:],batch_loss_window[:-1])):
				self.learning_rate = self.learning_rate * self.increase_factor
				did_increase_rate = True
				if self.verbose:
					print('Learning rate increased to %.2e'
						  % self.learning_rate)

		return did_increase_rate

	def _conditional_decrease_rate(self):
		'''Decreases the learning rate if the most recent loss is worse than
		all of the previous n loss values, and if no learning rate changes
		have been made in the last n steps, where n=min_steps_per_decrease.
		Args:
			None.
		Returns:
			A bool indicating whether the learning rate was decreased.
		'''

		did_decrease_rate = False
		n = self.min_steps_per_decrease

		if self.do_decrease_rate and self.step>=(self.step_last_update + n):

			batch_loss_window = self.loss_log[-(1+n):]
			lastBatchLoss = batch_loss_window[-1]

			if all(np.greater(batch_loss_window[-1],batch_loss_window[:-1])):
				self.learning_rate = self.learning_rate * self.decrease_factor
				did_decrease_rate = True
				if self.verbose:
					print('Learning rate decreased to %.2e'
						  % self.learning_rate)

		return did_decrease_rate

	def test(self, bias=0.0, fig=None):
		''' Generates and plots an adaptive learning rate schedule based on a
		loss function that is a 1-dimensional biased random walk. This can be
		used as a zero-th order analysis of hyperparameter settings,
		understanding that in a realistic optimization setting, the loss will
		depend highly on the learning rate (such dependencies are not included
		in this simulation).
		Args:
			bias: A float specifying the bias of the random walk used to
			simulate loss values.
		Returns:
			None.
		'''

		save_step = min(1000, self.max_n_steps/4)
		save_dir = '/tmp/'

		# Simulation 1
		loss = 0.
		loss_history = []
		rate_history = []
		while not self.is_finished():
			if self.step == save_step:
				print('Step %d: saving so that we can test restore()' %
					self.step)
				self.save(save_dir)

			loss = loss + bias + npr.randn()
			rate_history.append(self.update(loss))
			loss_history.append(loss)

			if np.mod(self.step, 100) == 0:
				print('Step %d...' % self.step)

		print('Step %d: simulation 1 complete.' % self.step)

		# Simlulation 2, tests restore(...)
		restored_alr = AdaptiveLearningRate()
		restored_alr.restore(save_dir)
		restored_rate_history = [np.nan] * save_step
		while not restored_alr.is_finished():
			# Use exactly the same loss values from the first simulation
			loss = loss_history[restored_alr.step]
			restored_rate_history.append(restored_alr.update(loss))

		print('Step %d: simulation 2 complete.' % restored_alr.step)

		diff = np.array(rate_history[save_step:]) - \
			np.array(restored_rate_history[save_step:])
		mean_abs_restore_error = np.mean(np.abs(diff))
		print('Avg abs diff between original and restored: %.3e' %
			mean_abs_restore_error)

		if fig is None:
			fig = plt.figure()

		ax1 = fig.add_subplot(2,1,1)
		ax1.plot(rate_history)
		ax1.plot(restored_rate_history, linestyle='--')
		ax1.set_yscale('log')
		ax1.set_ylabel('Learning rate')

		ax2 = fig.add_subplot(2,1,2)
		ax2.plot(loss_history)
		ax2.set_ylabel('Simulated loss')
		ax2.set_xlabel('Step')

		fig.show()


'''
FixedPoints Class
Supports FixedPointFinder
Written using Python 2.7.12 and TensorFlow 1.10.
@ Matt Golub, October 2018.
Please direct correspondence to mgolub@stanford.edu.
'''


import pdb
import numpy as np
import pickle


class FixedPoints(object):
    '''
    A class for storing fixed points and associated data.
    '''

    ''' List of class attributes that represent data corresponding to fixed
    points. All of these refer to Numpy arrays with axis 0 as the batch
    dimension. Thus, each is concatenatable using np.concatenate(..., axis=0).
    '''
    _data_attrs = [
            'xstar',
            'x_init',
            'inputs',
            'F_xstar',
            'qstar',
            'dq',
            'n_iters',
            'J_xstar',
            'eigval_J_xstar',
            'eigvec_J_xstar',
            'is_stable',
            'cond_id']

    ''' List of class attributes that apply to all fixed points
    (i.e., these are not indexed per fixed point). '''
    _nonspecific_attrs = [
        'dtype',
        'dtype_complex',
        'tol_unique',
        'verbose',
        'do_alloc_nan']

    def __init__(self,
                 xstar=None, # Fixed-point specific data
                 x_init=None,
                 inputs=None,
                 F_xstar=None,
                 qstar=None,
                 dq=None,
                 n_iters=None,
                 J_xstar=None,
                 eigval_J_xstar=None,
                 eigvec_J_xstar=None,
                 is_stable=None,
                 cond_id=None,
                 n=None,
                 n_states=None,
                 n_inputs=None, # Non-specific data
                 do_alloc_nan=False,
                 tol_unique=1e-3,
                 dtype=np.float32,
                 dtype_complex=np.complex64,
                 verbose=False):
        '''
        Initializes a FixedPoints object with all input arguments as class
        properties.

        Optional args:

            xstar: [n x n_states] numpy array with row xstar[i, :]
            specifying an the fixed point identified from x_init[i, :].
            Default: None.

            x_init: [n x n_states] numpy array with row x_init[i, :]
            specifying the initial state from which xstar[i, :] was optimized.
            Default: None.

            inputs: [n x n_inputs] numpy array with row inputs[i, :]
            specifying the input to the RNN during the optimization of
            xstar[i, :]. Default: None.

            F_xstar: [n x n_states] numpy array with F_xstar[i, :]
            specifying RNN state after transitioning from the fixed point in
            xstar[i, :]. If the optimization succeeded (e.g., to 'tol') and
            identified a stable fixed point, the state should not move
            substantially from the fixed point (i.e., xstar[i, :] should be
            very close to F_xstar[i, :]). Default: None.

            qstar: [n,] numpy array with qstar[i] containing the
            optimized objective (1/2)(x-F(x))^T(x-F(x)), where
            x = xstar[i, :]^T and F is the RNN transition function (with the
            specified constant inputs). Default: None.

            dq: [n,] numpy array with dq[i] containing the absolute
            difference in the objective function after (i.e., qstar[i]) vs
            before the final gradient descent step of the optimization of
            xstar[i, :]. Default: None.

            n_iters: [n,] numpy array with n_iters[i] as the number of
            gradient descent iterations completed to yield xstar[i, :].
            Default: None.

            J_xstar: [n x n_states x n_states] numpy array with
            J_xstar[i, :, :] containing the Jacobian of the RNN state
            transition function at fixed point xstar[i, :]. Default: None,
            which results in an appropriately sized numpy array of NaNs.
            Default: None.

            eigval_J_xstar: [n x n_states] numpy array with
            eigval_J_xstar[i, :] containing the eigenvalues of
            J_xstar[i, :, :].

            eigvec_J_xstar: [n x n_states x n_states] numpy array with
            eigvec_J_xstar[i, :, :] containing the eigenvectors of
            J_xstar[i, :, :].

            is_stable: [n,] numpy array with is_stable[i] indicating as bool
            whether xstar[i] is a stable fixed point.

            do_alloc_nan: Bool indicating whether to initialize all data
            attributes (all optional args above) as NaN-filled numpy arrays.
            Default: False.

                If True, n, n_states and n_inputs must be provided. These
                values are otherwise ignored:

                n: Positive int specifying the number of fixed points to
                allocate space for.

                n_states: Positive int specifying the dimensionality of the
                network state (a.k.a. the number of hidden units).

                n_inputs: Positive int specifying the dimensionality of the
                network inputs.

            tol_unique: Positive scalar specifying the numerical precision
            required to label two fixed points as being unique from one
            another. Two fixed points are considered unique if the 2-norm of
            the difference between their concatenated (xstar, inputs) is
            greater than this tolerance. Default: 1e-3.

            dtype: Data type for representing all of the object's data.
            Default: numpy.float32.

            cond_id: [n,] numpy array with cond_id[i] indicating the condition ID corresponding to inputs[i].

            verbose: Bool indicating whether to print status updates.

        Note:
            xstar, x_init, inputs, F_xstar, and J_xstar are all numpy arrays,
            regardless of whether that type is consistent with the state type
            of the rnncell from which they originated (i.e., whether or not
            the rnncell is an LSTM). This design decision reflects that a
            Jacobian is most naturally expressed as a single matrix (as
            opposed to a collection of matrices representing interactions
            between LSTM hidden and cell states). If one requires state
            representations as type LSTMStateCell, use
            FixedPointFinder._convert_to_LSTMStateTuple.

        Returns:
            None.

        '''

        # These apply to all fixed points
        # (one value each, rather than one value per fixed point).
        self.tol_unique = tol_unique
        self.dtype = dtype
        self.dtype_complex = dtype_complex
        self.do_alloc_nan = do_alloc_nan
        self.verbose = verbose

        if do_alloc_nan:

            if n is None:
                raise ValueError('n must be provided if '
                                 'do_alloc_nan == True.')
            if n_states is None:
                raise ValueError('n_states must be provided if '
                                 'do_alloc_nan == True.')
            if n_inputs is None:
                raise ValueError('n_inputs must be provided if '
                                 'do_alloc_nan == True.')

            self.n = n
            self.n_states = n_states
            self.n_inputs = n_inputs

            self.xstar = self._alloc_nan((n, n_states))
            self.x_init = self._alloc_nan((n, n_states))
            self.inputs = self._alloc_nan((n, n_inputs))
            self.F_xstar = self._alloc_nan((n, n_states))
            self.qstar = self._alloc_nan((n))
            self.dq = self._alloc_nan((n))
            self.n_iters = self._alloc_nan((n))
            self.J_xstar = self._alloc_nan((n, n_states, n_states))

            self.eigval_J_xstar = self._alloc_nan(
                (n, n_states), dtype=dtype_complex)
            self.eigvec_J_xstar = self._alloc_nan(
                (n, n_states, n_states), dtype=dtype_complex)

            # not forcing dtype to bool yet, since np.bool(np.nan) is True,
            # which could be misinterpreted as a valid value.
            self.is_stable = self._alloc_nan((n))

            self.cond_id = self._alloc_nan((n))

        else:
            if xstar is not None:
                self.n, self.n_states = xstar.shape
            elif x_init is not None:
                self.n, self.n_states = x_init.shape
            elif F_xstar is not None:
                self.n, self.n_states = F_xstar.shape
            elif J_xstar is not None:
                self.n, self.n_states, _ = J_xstar.shape
            else:
                self.n = None
                self.n_states = None

            if inputs is not None:
                self.n_inputs = inputs.shape[1]
                if self.n is None:
                    self.n = inputs.shape[0]
            else:
                self.n_inputs = None

            self.xstar = xstar
            self.x_init = x_init
            self.inputs = inputs
            self.F_xstar = F_xstar
            self.qstar = qstar
            self.dq = dq
            self.n_iters = n_iters
            self.J_xstar = J_xstar
            self.eigval_J_xstar = eigval_J_xstar
            self.eigvec_J_xstar = eigvec_J_xstar
            self.is_stable = is_stable
            self.cond_id = cond_id

        self.assert_valid_shapes()

    def __setitem__(self, index, fps):
        '''Implements the assignment operator.

        All compatible data from fps are copied. This excludes tol_unique,
        dtype, n, n_states, and n_inputs, which retain their original values.

        Usage:
            fps_to_be_partially_overwritten[index] = fps
        '''

        assert isinstance(fps, FixedPoints),\
            ('fps must be a FixedPoints object but was %s.' % type(fps))

        if isinstance(index, int):
            # Force the indexing that follows to preserve numpy array ndim
            index = range(index, index+1)

        manual_data_attrs = ['eigval_J_xstar', 'eigvec_J_xstar', 'is_stable']

        # This block added for testing 9/17/20 (replaces commented code below)
        for attr_name in self._data_attrs:
            if attr_name not in manual_data_attrs:
                attr = getattr(self, attr_name)
                if attr is not None:
                    attr[index] = getattr(fps, attr_name)

        ''' Previous version of block above:

        if self.xstar is not None:
            self.xstar[index] = fps.xstar

        if self.x_init is not None:
            self.x_init[index] = fps.x_init

        if self.inputs is not None:
            self.inputs[index] = fps.inputs

        if self.F_xstar is not None:
            self.F_xstar[index] = fps.F_xstar

        if self.qstar is not None:
            self.qstar[index] = fps.qstar

        if self.dq is not None:
            self.dq[index] = fps.dq

        if self.J_xstar is not None:
            self.J_xstar[index] = fps.J_xstar
        '''

        # This manual handling no longer seems necessary, but I'll save that
        # change and testing for a rainy day.
        if self.has_decomposed_jacobians:
            self.eigval_J_xstar[index] = fps.eigval_J_xstar
            self.eigvec_J_xstar[index] = fps.eigvec_J_xstar
            self.is_stable[index] = fps.is_stable

    def __getitem__(self, index):
        '''Indexes into a subset of the fixed points and their associated data.

        Usage:
            fps_subset = fps[index]

        Args:
            index: a slice object for indexing into the FixedPoints data.

        Returns:
            A FixedPoints object containing a subset of the data from the
            current FixedPoints object, as specified by index.
        '''

        if isinstance(index, int):
            # Force the indexing that follows to preserve numpy array ndim
            index = range(index, index+1)

        kwargs = self._nonspecific_kwargs
        manual_data_attrs = ['eigval_J_xstar', 'eigvec_J_xstar', 'is_stable']

        for attr_name in self._data_attrs:

            attr_val = getattr(self, attr_name)

            # This manual handling no longer seems necessary, but I'll save
            # that change and testing for a rainy day.
            if attr_name in manual_data_attrs:
                if self.has_decomposed_jacobians:
                    indexed_val = self._safe_index(attr_val, index)
                else:
                    indexed_val = None
            else:
                indexed_val = self._safe_index(attr_val, index)

            kwargs[attr_name] = indexed_val

        indexed_fps = FixedPoints(**kwargs)

        return indexed_fps

    def __len__(self):
        '''Returns the number of fixed points stored in the object.'''
        return self.n

    def __contains__(self, fp):
        '''Checks whether a specified fixed point is contained in the object.

        Args:
            fp: A FixedPoints object containing exactly one fixed point.

        Returns:
            bool indicating whether any fixed point matches fp.
        '''

        idx = self.find(fp)

        return idx.size > 0

    def get_unique(self):
        '''Identifies unique fixed points. Among duplicates identified,
        this keeps the one with smallest qstar.

        Args:
            None.

        Returns:
            A FixedPoints object containing only the unique fixed points and
            their associated data. Uniqueness is determined down to tol_unique.
        '''
        assert (self.xstar is not None),\
            ('Cannot find unique fixed points because self.xstar is None.')

        if self.inputs is None:
            data_nxd = self.xstar
        else:
            data_nxd = np.concatenate((self.xstar, self.inputs), axis=1)

        idx_keep = []
        idx_checked = np.zeros(self.n, dtype=np.bool)
        for idx in range(self.n):

            if idx_checked[idx]:
                # If this FP matched others, we've already determined which
                # of those matching FPs to keep. Repeating would simply
                # identify the same FP to keep.
                continue

            # Don't compare against FPs we've already checked
            idx_check = np.where(~idx_checked)[0]
            fps_check = self[idx_check] # only check against these FPs
            idx_idx_check = fps_check.find(self[idx]) # indexes into fps_check
            idx_match = idx_check[idx_idx_check] # indexes into self

            if len(idx_match)==1:
                # Only matches with itself
                idx_keep.append(idx)
            else:
                qstars_match = self.qstar[idx_match]
                idx_candidate = idx_match[np.argmin(qstars_match)]
                idx_keep.append(idx_candidate)
                idx_checked[idx_match] = True

        return self[idx_keep]

    def transform(self, U, offset=0.):
        ''' Apply an affine transformation to the state-space representation.
        This may be helpful for plotting fixed points in a given linear
        subspace (e.g., PCA or an RNN readout space).


        Args:
            U: shape (n_states, k) numpy array projection matrix.

            offset (optional): shape (k,) numpy translation vector. Default: 0.

        Returns:
            A FixedPoints object.
        '''
        kwargs = self.kwargs

        # These are all transformed. All others are not.
        for attr_name in ['xstar', 'x_init', 'F_xstar']:
            kwargs[attr_name] = np.matmul(getattr(self, attr_name), U) + offset

        if self.has_decomposed_jacobians:
            kwargs['eigval_J_xstar'] = self.eigval_J_xstar
            kwargs['eigvec_J_xstar'] = \
                np.matmul(U.T, self.eigvec_J_xstar) + offset

        transformed_fps = FixedPoints(**kwargs)

        return transformed_fps

    def find(self, fp):
        '''Searches in the current FixedPoints object for matches to a
        specified fixed point. Two fixed points are defined as matching
        if the 2-norm of the difference between their concatenated (xstar,
        inputs) is within tol_unique).

        Args:
            fp: A FixedPoints object containing exactly one fixed point.

        Returns:
            shape (n_matches,) numpy array specifying indices into the current
            FixedPoints object where matches to fp were found.
        '''

        # If not found or comparison is impossible (due to type or shape),
        # follow convention of np.where and return an empty numpy array.
        result = np.array([], dtype=int)

        if isinstance(fp, FixedPoints):
            if fp.n_states == self.n_states and fp.n_inputs == self.n_inputs:

                if self.inputs is None:
                    self_data_nxd = self.xstar
                    arg_data_nxd = fp.xstar
                else:
                    self_data_nxd = np.concatenate(
                        (self.xstar, self.inputs), axis=1)
                    arg_data_nxd = np.concatenate(
                        (fp.xstar, fp.inputs), axis=1)

                norm_diffs_n = np.linalg.norm(
                    self_data_nxd - arg_data_nxd, axis=1)

                result = np.where(norm_diffs_n <= self.tol_unique)[0]

        return result

    def update(self, new_fps):
        ''' Combines the entries from another FixedPoints object into this
        object.

        Args:
            new_fps: a FixedPoints object containing the entries to be
            incorporated into this FixedPoints object.

        Returns:
            None

        Raises:
            AssertionError if the non-fixed-point specific attributes of
            new_fps do not match those of this FixedPoints object.

            AssertionError if any data attributes are found in one but not both
            FixedPoints objects (especially relevant for decomposed Jacobians).

            AssertionError if the updated object has inconsistent data shapes.
        '''

        self._assert_matching_nonspecific_attrs(self, new_fps)

        for attr_name in self._data_attrs:

            this_has = hasattr(self, attr_name)
            that_has = hasattr(new_fps, attr_name)

            assert this_has == that_has,\
                ('One but not both FixedPoints objects have %s. '
                 'FixedPoints.update does not currently support this '
                 'configuration.' % attr_name)

            if this_has and that_has:
                cat_attr = np.concatenate(
                    (getattr(self, attr_name),
                    getattr(new_fps, attr_name)),
                    axis=0)
                setattr(self, attr_name, cat_attr)

        self.n = self.n + new_fps.n
        self.assert_valid_shapes()

    def decompose_jacobians(self, do_batch=True, str_prefix=''):
        '''Adds the following fields to the FixedPoints object:

        eigval_J_xstar: [n x n_states] numpy array with eigval_J_xstar[i, :]
        containing the eigenvalues of J_xstar[i, :, :].

        eigvec_J_xstar: [n x n_states x n_states] numpy array containing with
        eigvec_J_xstar[i, :, :] containing the eigenvectors of
        J_xstar[i, :, :].

        Args:
            do_batch (optional): bool indicating whether to perform a batch
            decomposition. This is typically faster as long as sufficient
            memory is available. If False, decompositions are performed
            one-at-a-time, sequentially, which may be necessary if the batch
            computation requires more memory than is available. Default: True.

            str_prefix (optional): String to be pre-pended to print statements.

        Returns:
            None.
        '''

        if self.has_decomposed_jacobians:
            print('%sJacobians have already been decomposed, '
                'not repeating.' % str_prefix)
            return

        n = self.n # number of FPs represented in this object
        n_states = self.n_states # dimensionality of each state

        if do_batch:
            # Batch eigendecomposition
            print('%sDecomposing Jacobians in a single batch.' % str_prefix)

            # Check for NaNs in Jacobians
            valid_J_idx = ~np.any(np.isnan(self.J_xstar), axis=(1,2))

            if np.all(valid_J_idx):
                # No NaNs, nothing to worry about.
                e_vals_unsrt, e_vecs_unsrt = np.linalg.eig(self.J_xstar)
            else:
                # Set eigen-data to NaN if there are any NaNs in the
                # corresponding Jacobian.
                e_vals_unsrt = self._alloc_nan(
                    (n, n_states), dtype=self.dtype_complex)
                e_vecs_unsrt = self._alloc_nan(
                    (n, n_states, n_states), dtype=dtype_complex)

                e_vals_unsrt[valid_J_idx], e_vecs_unsrt[valid_J_idx] = \
                    np.linalg.eig(self.J_xstar[valid_J_idx])

        else:
            print('%sDecomposing Jacobians one-at-a-time.' % str_prefix)
            e_vals = []
            e_vecs = []
            for J in self.J_xstar:

                if np.any(np.isnan(J)):
                    e_vals_i = self._alloc_nan((n_states,))
                    e_vecs_i = self._alloc_nan((n_states, n_states))
                else:
                    e_vals_i, e_vecs_i = np.linalg.eig(J)

                e_vals.append(np.expand_dims(e_vals_i, axis=0))
                e_vecs.append(np.expand_dims(e_vecs_i, axis=0))

            e_vals_unsrt = np.concatenate(e_vals, axis=0)
            e_vecs_unsrt = np.concatenate(e_vecs, axis=0)

        print('%sSorting by Eigenvalue magnitude.' % str_prefix)
        # For each FP, sort eigenvectors by eigenvalue magnitude
        # (decreasing order).
        mags_unsrt = np.abs(e_vals_unsrt) # shape (n,)
        sort_idx = np.argsort(mags_unsrt)[:,::-1]

        # Apply the sort
        # There must be a faster way, but I'm too lazy to find it at the moment
        self.eigval_J_xstar = \
            self._alloc_nan((n, n_states), dtype=self.dtype_complex)
        self.eigvec_J_xstar = \
            self._alloc_nan((n, n_states, n_states), dtype=self.dtype_complex)
        self.is_stable = np.zeros(n, dtype=np.bool)

        for k in range(n):
            sort_idx_k = sort_idx[k]
            e_vals_k = e_vals_unsrt[k][sort_idx_k]
            e_vecs_k = e_vecs_unsrt[k][:, sort_idx_k]
            self.eigval_J_xstar[k] = e_vals_k
            self.eigvec_J_xstar[k] = e_vecs_k

            # For stability, need only to look at the leading eigenvalue
            self.is_stable[k] = np.abs(e_vals_k[0]) < 1.0

        self.assert_valid_shapes()

    def save(self, save_path):
        '''Saves all data contained in the FixedPoints object.

        Args:
            save_path: A string containing the path at which to save
            (including directory, filename, and arbitrary extension).

        Returns:
            None.
        '''
        if self.verbose:
            print('Saving FixedPoints object.')

        self.assert_valid_shapes()

        file = open(save_path,'w')
        file.write(cPickle.dumps(self.__dict__))
        file.close()

    def restore(self, restore_path):
        '''Restores data from a previously saved FixedPoints object.

        Args:
            restore_path: A string containing the path at which to find a
            previously saved FixedPoints object (including directory, filename,
            and extension).

        Returns:
            None.
        '''
        if self.verbose:
            print('Restoring FixedPoints object.')
        file = open(restore_path,'r')
        restore_data = file.read()
        file.close()
        self.__dict__ = cPickle.loads(restore_data)

        # Hack to bridge between different versions of saved data
        if not hasattr(self, 'do_alloc_nan'):
            self.do_alloc_nan = False

        self.assert_valid_shapes()

    def print_summary(self):
        '''Prints a summary of the fixed points.

        Args:
            None.

        Returns:
            None.
        '''

        print('\nThe q function at the fixed points:')
        print(self.qstar)

        print('\nChange in the q function from the final iteration '
              'of each optimization:')
        print(self.dq)

        print('\nNumber of iterations completed for each optimization:')
        print(self.n_iters)

        print('\nThe fixed points:')
        print(self.xstar)

        print('\nThe fixed points after one state transition:')
        print(self.F_xstar)
        print('(these should be very close to the fixed points)')

        if self.J_xstar is not None:
            print('\nThe Jacobians at the fixed points:')
            print(self.J_xstar)

    def print_shapes(self):
        ''' Prints the shapes of the data attributes of the fixed points.

        Args:
            None.

        Returns:
            None.
        '''

        for attr_name in FixedPoints._data_attrs:
            attr = getattr(self, attr_name)
            print('%s: %s' % (attr_name, str(attr.shape)))


    def assert_valid_shapes(self):
        ''' Checks that all data attributes reflect the same number of fixed
        points.

        Raises:
            AssertionError if any non-None data attribute does not have
            .shape[0] as self.n.
        '''
        n = self.n
        for attr_name in FixedPoints._data_attrs:
            data = getattr(self, attr_name)
            if data is not None:
                assert data.shape[0] == self.n,\
                    ('Detected %d fixed points, but %s.shape is %s '
                    '(shape[0] should be %d' %
                    (n, attr_name, str(data.shape), n))

    @staticmethod
    def concatenate(fps_seq):
        ''' Join a sequence of FixedPoints objects.

        Args:
            fps_seq: sequence of FixedPoints objects. All FixedPoints objects
            must have the following attributes in common:
                n_states
                n_inputs
                has_decomposed_jacobians

        Returns:
            A FixedPoints objects containing the concatenated FixedPoints data.
        '''

        assert len(fps_seq) > 0, 'Cannot concatenate empty list.'
        FixedPoints._assert_matching_nonspecific_attrs(fps_seq)

        kwargs = {}

        for attr_name in FixedPoints._nonspecific_attrs:
            kwargs[attr_name] = getattr(fps_seq[0], attr_name)

        for attr_name in FixedPoints._data_attrs:
            if all((hasattr(fps, attr_name) for fps in fps_seq)):

                cat_list = [getattr(fps, attr_name) for fps in fps_seq]

                if all([l is None for l in cat_list]):
                    cat_attr = None
                elif any([l is None for l in cat_list]):
                    # E.g., attempting to concat cond_id when it exists for
                    # some fps but not for others. Better handling of this
                    # would be nice. And yes, this would catch the all above,
                    # but I'm keeping these cases separate to facilitate an
                    # eventual refinement.
                    cat_attr = None
                else:
                    cat_attr = np.concatenate(cat_list, axis=0)

                kwargs[attr_name] = cat_attr

        return FixedPoints(**kwargs)

    @property
    def is_single_fixed_point(self):
        return self.n == 1

    @property
    def has_decomposed_jacobians(self):

        if not hasattr(self, 'eigval_J_xstar'):
            return False

        return self.eigval_J_xstar is not None

    @property
    def kwargs(self):
        ''' Returns dict of keyword arguments necessary for reinstantiating a
        (shallow) copy of this FixedPoints object, i.e.,

        fp_copy  = FixedPoints(**fp.kwargs)
        '''

        kwargs = self._nonspecific_kwargs

        for attr_name in self._data_attrs:
            kwargs[attr_name] = getattr(self, attr_name)

        return kwargs

    def _alloc_nan(self, shape, dtype=None):
        '''Returns a nan-filled numpy array.

        Args:
            shape: int or tuple representing the shape of the desired numpy
            array.

        Returns:
            numpy array with the desired shape, filled with NaNs.

        '''
        if dtype is None:
            dtype = self.dtype

        result = np.zeros(shape, dtype=dtype)
        result.fill(np.nan)
        return result

    @staticmethod
    def _assert_matching_nonspecific_attrs(fps_seq):

        for attr_name in FixedPoints._nonspecific_attrs:
            items = [getattr(fps, attr_name) for fps in fps_seq]
            for item in items:
                assert item == items[0],\
                    ('Cannot concatenate FixedPoints because of mismatched %s '
                     '(%s is not %s)' %
                     (attr_name, str(items[0]), str(item)))

    @staticmethod
    def _safe_index(x, idx):
        '''Safe method for indexing into a numpy array that might be None.

        Args:
            x: Either None or a numpy array.

            idx: Positive int or index-compatible argument for indexing into x.

        Returns:
            Self explanatory.

        '''
        if x is None:
            return None
        else:
            return x[idx]


    @property
    def _nonspecific_kwargs(self):
        # These are not specific to individual fixed points.
        # Thus, simple copy, no indexing required
        return {
            'dtype': self.dtype,
            'tol_unique': self.tol_unique
            }