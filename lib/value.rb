# Stores a single scalar value and its gradient.
class Value
  attr_reader :data, :children, :operator
  attr_accessor :_backward, :grad

  def initialize(data, operator: nil, children: [], backward: -> {})
    @data = data
    @grad = 0.0
    @_backward = backward
    @children = children
    @operator = operator
  end

  def +(other)
    out = Value.new data + other.data, operator: :+, children: [self, other]
    out._backward = lambda {
      @grad += out.grad
      other.grad += out.grad
    }
    out
  end

  def *(other)
    out = Value.new data * other.data, operator: :*, children: [self, other]
    out._backward = lambda {
      @grad += other.data * out.grad
      other.grad += data * out.grad
    }
    out
  end

  def **(other)
    out = Value.new data**other, operator: :"**#{other}", children: [self]
    out._backward = -> { @grad += (other * (data**(other - 1))) * out.grad }
    out
  end

  def -@ = self * -1
  def -(other) = self + -other
  def /(other) = self * (other**-1)

  def relu
    out = Value.new data.negative? ? 0 : data, operator: :relu, children: [self]
    out._backward = -> { @grad += out.positive? ? out.grad : 0 }
    out
  end

  def backward
    @grad = 1.0
    topo_sort.reverse_each { _1._backward.call }
  end

  def topo_sort(visited = Set.new, sorted_values = [])
    return if visited.include? self

    visited << self
    children.each { _1.topo_sort visited, sorted_values }
    sorted_values << self
  end

  def inspect
    "Value(data=#{data}, grad=#{grad})"
  end
end
