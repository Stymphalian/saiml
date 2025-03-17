from collections import defaultdict
import numpy as np
import math

class ClassWithID:
    _GLOBAL_ID = 1
    def __init__(self):
        self.id = ClassWithID._GLOBAL_ID
        ClassWithID._GLOBAL_ID += 1

class Operator:
    def __init__(self):
        pass
    def value(self, *args, **kwargs):
        pass
    def gradient(self, respectTo, *args, **kwargs):
        pass

class Add(Operator):
    def value(self, *args, **kwargs):
        return args[0].value() + args[1].value()

    def gradient(self, respectTo, *args, **kwargs):
        return 1
    
class Sub(Operator):
    def value(self, *args, **kwargs):
        return args[0].value() - args[1].value()
    def gradient(self, respectTo, *args, **kwargs):
        if respectTo == args[0]:
            return 1
        elif respectTo == args[1]:
            return -1
        return 0

class Mult(Operator):
    def value(self, *args, **kwargs):
        return args[0].value() * args[1].value()
    def gradient(self, respectTo, *args, **kwargs):
        if respectTo == args[0]:
            return args[1].value()
        elif respectTo == args[1]:
            return args[0].value()
        return 0
    
class Div(Operator):
    def value(self, *args, **kwargs):
        return args[0].value() / args[1].value()
    def gradient(self, respectTo, *args, **kwargs):
        if respectTo == args[0]:
            return 1 / args[1].value()
        elif respectTo == args[1]:
            return -args[0].value() / args[1].value() ** 2
        
class Log(Operator):
    def value(self, *args, **kwargs):
        return math.log(args[0].value())
    def gradient(self, respectTo, *args, **kwargs):
        if respectTo == args[0]:
            return 1 / args[0].value()
        return 0
        
class Sin(Operator):
    def value(self, *args, **kwargs):
        return math.sin(args[0].value())
    def gradient(self, respectTo, *args, **kwargs):
        if respectTo == args[0]:
            return math.cos(args[0].value())
        return 0
    
class Constant(Operator):
    def __init__(self, value):
        super().__init__
        self.val = value
    def value(self, *args, **kwargs):
        return self.val
    def gradient(self, respectTo, *args, **kwargs):
        return 0

class Node(ClassWithID):
    def __init__(self, parents=[], operator=None):
        super().__init__()
        self.operator = operator
        self.parents = parents
        self.cached_value = None

    def value(self):
        if self.cached_value is None:
            self.cached_value = self.operator.value(*self.parents)
        return self.cached_value

    def gradient(self, respectTo):
        return self.operator.gradient(respectTo, *self.parents)
    
    def __add__(self, y):
        return Node([self,y], Add())

    def __sub__(self, y):
        return Node([self,y], Sub())
    
    def __mul__(self, y):
        return Node([self, y], Mult())
    
    def __truediv__(self, y):
        return Node([self, y], Div())

    def grad(self, parent):
        if parent not in self.parents:
            return 0.0
        

def constant(value):
    return Node([], Constant(value))

def log(x: Node):
    return Node([x], Log())

def add(x: Node, y: Node):
    return Node([x, y], Add())

def sub(x: Node, y: Node):
    return Node([x, y], Sub())

def mult(x: Node, y: Node):
    return Node([x, y], Mult())

def sin(x: Node):
    return Node([x], Sin())

def toposort(end_node):
    """
    Topologically sort the DAG starting from the end_node.
    This is used during the traversal of the computational graph to ensure
    we visit every node such that all child dependencies are computed before
    processing any of the parents.

    https://en.wikipedia.org/wiki/Topological_sorting
    https://github.com/mattjj/autodidact/blob/master/autograd/util.py
    """

    # Make a dictionary counting the number of edges connecting to the node
    # We will use this in the toposort sort process to determine which nodes
    # to add to the search space
    child_counts = {}
    stack = [end_node]
    while stack:
        node = stack.pop()
        if node in child_counts:
            child_counts[node] += 1
        else:
            child_counts[node] = 1
            stack.extend(node.parents)

    # Start with the end node.
    # The subsequent nodes to output should only occur once it has "become"
    # a childless nodes (ie. we have already visited all of the dependencies)
    childless_nodes = [end_node]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for parent in node.parents:
            if child_counts[parent] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent] -= 1

def gradient(node: Node):
    node_to_grads = defaultdict(list)
    node_to_grads[node.id] = [1.0]
    
    for v in toposort(node):
        dy_dv = sum(node_to_grads[v.id])
        for p in v.parents:
            dv_dp = v.gradient(p)
            dy_dp = dy_dv * dv_dp
            node_to_grads[p.id].append(dy_dp)
    return node_to_grads

def value(node: Node):
    return node.value()