import tensorflow as tf
from smartink.util.utils import err_unknown_type


class LearningRateFactory(object):
  """Generates a learning rate scheduler object."""
  
  def __init__(self, lr_config):
    self.lr_type = lr_config["name"]
  
  @classmethod
  def get(cls, config):
    lr_type = config["name"]
    if lr_type == "exponential":
      return ExponentialDecay(**config)
    elif lr_type == "sketch_rnn":
      return SketchRnnDecay(**config)
    elif lr_type == "transformer":
      return TransformerDecay(**config)
    else:
      err_unknown_type(lr_type)


class ExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A LearningRateSchedule that follows bounded exponential decay schedule."""
  
  def __init__(
      self,
      initial_learning_rate,
      decay_steps=1000,
      decay_rate=0.96,
      staircase=False,
      min_learning_rate=0.,
      name=None,
      **kwargs):
    super(ExponentialDecay, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.min_learning_rate = min_learning_rate
    self.decay_steps = float(decay_steps)
    self.decay_rate = float(decay_rate)
    self.staircase = staircase
    self.name = name
    
    self.unbounded_exp_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate, staircase=staircase,
        name=name)
  
  def __call__(self, step):
    return tf.maximum(self.unbounded_exp_lr(step), self.min_learning_rate)
  
  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "min_learning_rate"    : self.min_learning_rate,
        "decay_steps"          : self.decay_steps,
        "decay_rate"           : self.decay_rate,
        "staircase"            : self.staircase,
        "name"                 : self.name
        }


class TransformerDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A LearningRateSchedule that follows transformer schedule."""
  
  def __init__(
      self,
      initial_learning_rate,
      d_model=64,
      warmup_steps=4000,
      min_learning_rate=0.,
      name=None,
      **kwargs):
    super(TransformerDecay, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.min_learning_rate = min_learning_rate
    self.d_model = float(d_model)
    self.warmup_steps = float(warmup_steps)
    self.name = name
  
  def __call__(self, step):
    float_global_step = tf.cast(step, tf.float32)
    
    arg1 = tf.math.rsqrt(float_global_step)
    arg2 = float_global_step*(self.warmup_steps**-1.5)
    return tf.math.rsqrt(self.d_model)*tf.math.minimum(arg1, arg2)
  
  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "min_learning_rate"    : self.min_learning_rate,
        "d_model"              : self.d_model,
        "warmup_steps"         : self.warmup_steps,
        "name"                 : self.name,
        }


class SketchRnnDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A LearningRateSchedule that follows sketch-rnn schedule."""
  
  def __init__(
      self,
      initial_learning_rate,
      decay_rate=0.9999,
      min_learning_rate=0.,
      name=None,
      **kwargs):
    super(SketchRnnDecay, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.min_learning_rate = min_learning_rate
    self.decay_rate = decay_rate
    self.name = name
  
  def __call__(self, step):
    float_global_step = tf.cast(step, tf.float32)
    return (self.initial_learning_rate - self.min_learning_rate)*self.decay_rate**float_global_step + self.min_learning_rate
  
  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "min_learning_rate"    : self.min_learning_rate,
        "decay_rate"           : self.decay_rate,
        "name"                 : self.name,
        }