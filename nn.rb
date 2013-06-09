class Neuron
  attr_accessor :weights, :inputs, :threshold, :error, :learning_rate, :position
  
  def initialize(num_inputs, position_in_layer)
    self.inputs = []
    self.weights = n_rand(num_inputs)
    self.threshold = 1
    self.position = position_in_layer
  end

  def n_rand(n)
    (0...n).map{ rand }
  end

  def output
    a = 1
    x = threshold + weighted_sum
    1.0 / (1 + Math::E**(-1 * a * x) )
  end

  def weighted_sum
    weights.zip(inputs).map{|x,y| x * y}.inject(&:+)
  end

  def train(e)
    delta_thresh = e * learning_rate
    self.threshold += delta_thresh

    self.inputs.each_with_index do |xi, i|
      self.weights[i] += delta_thresh * xi
    end
  end
end

class HiddenNeuron < Neuron
  def initialize(num_inputs, position)
    self.learning_rate = 0.15
    super(num_inputs, position)
  end

  def error(output_layer)
    z = output
    g = output_layer.map{|o| o.weights[position] * o.error}.inject(&:+)
    z * (1 - z) * g
  end
end

class OutputNeuron < Neuron
  def initialize(num_inputs, position)
    self.learning_rate = 0.2
    super(num_inputs, position)
  end
    
  #def output
  # TODO: thought I would just return weighted_sum + threshold for linear output for output nodes, but doesn't seem to work
  #end

  def error(desired=nil)
    unless desired.nil?
      o = output
      d = desired
      @cached_error = o * (1 - o) * (d - o)
    end

    @cached_error
  end
end

class Network
  attr_accessor :hidden_layer, :output_layer

  def initialize(num_inputs = 3, num_hidden = 5, num_outputs = 1)
    self.hidden_layer = (0...num_hidden).map{ |i| HiddenNeuron.new(num_inputs, i) }
    self.output_layer = (0...num_outputs).map{ |j| OutputNeuron.new(num_hidden, j) }
  end

  def forward(inputs)
    hidden_layer.each do |h|
      h.inputs = inputs
    end

    output_layer.each do |o|
      o.inputs = hidden_layer.map(&:output)
    end

    output_layer.map(&:output)
  end

  def backward(desired_result)
    output_layer.each do |o|
      o.train(o.error(desired_result))
    end

    hidden_layer.each do |h|
      h.train(h.error(output_layer))
    end
  end

  def train(inputs, desired_result)
    forward(inputs)
    backward(desired_result)
  end
end

def random_cases(n)
  (0...n).map{ z = [rand / 3.0, rand / 3.0, rand / 3.0]; [z, z.inject(&:+) ]}
end

def run(num_examples=100_000, num_tests=1_000)
  nn = Network.new

  examples = random_cases(num_examples)
  
  examples.each do |e|
    nn.train(e[0], e[1])
  end

  tests = random_cases(num_tests)

  error = 0.0
  tests.each do |test|
    d = test[1]
    a = nn.forward(test[0])[0]
    error += (d - a) ** 2
  end

  puts "sqrt(Avg Sq. Err): #{( error /tests.size ) ** 0.5}"
end

run
