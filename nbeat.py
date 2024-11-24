import tensorflow as tf
from tensorflow.keras import layers

# Create NBeatsBlock custom layer 
class NBeatsBlock(layers.Layer):
  def __init__(self, # the constructor takes all the hyperparameters for the layer
               input_size: int,
               theta_size: int,
               horizon: int,
               n_neurons: int,
               n_layers: int,
               **kwargs): # the **kwargs argument takes care of all of the arguments for the parent class (input_shape, trainable, name)
    super().__init__(**kwargs)
    self.input_size = input_size
    self.theta_size = theta_size
    self.horizon = horizon
    self.n_neurons = n_neurons
    self.n_layers = n_layers
    # Block contains stack of 4 fully connected layers each has ReLU activation
    self.hidden = [layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
    # Output of block is a theta layer with linear activation
    self.theta_layer = layers.Dense(theta_size, activation="linear", name="theta")
  def call(self, inputs): # the call method is what runs when the layer is called 
    x = inputs 
    for layer in self.hidden: # pass inputs through each hidden layer 
      x = layer(x)
    theta = self.theta_layer(x) 
    # Output the backcast and forecast from theta
    backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
    return backcast, forecast

def buildNBeatLayers(input_size: int, theta_size: int, horizon: int, n_neurons: int, n_layers: int, n_stacks: int, upper_layer):
  # Setup N-BEATS Block layer
  nbeats_block_layer = NBeatsBlock(input_size=input_size,
                                  theta_size=theta_size,
                                  horizon=horizon,
                                  n_neurons=n_neurons,
                                  n_layers=n_layers,
                                  name="InitialBlock")
  # Create initial backcast and forecast input (backwards predictions are referred to as residuals in the paper)
  backcast, forecast = nbeats_block_layer(upper_layer)
  # Add in subtraction residual link, thank you to: https://github.com/mrdbourke/tensorflow-deep-learning/discussions/174 
  residuals = layers.subtract([upper_layer, backcast], name=f"nbeat_subtract_00") 
  # Create stacks of blocks
  for i, _ in enumerate(range(n_stacks-1)): # first stack is already creted in (3)
    # Use the NBeatsBlock to calculate the backcast as well as block forecast
    backcast, block_forecast = NBeatsBlock(
        input_size=input_size,
        theta_size=theta_size,
        horizon=horizon,
        n_neurons=n_neurons,
        n_layers=n_layers,
        name=f"NBeatsBlock_{i}"
    )(residuals) # pass it in residuals (the backcast)
    # Create the double residual stacking
    residuals = layers.subtract([residuals, backcast], name=f"nbeat_subtract_{i}") 
    forecast = layers.add([forecast, block_forecast], name=f"nbeat_add_{i}")
  return forecast