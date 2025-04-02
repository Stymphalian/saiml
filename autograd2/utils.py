
from collections import defaultdict
import numpy as np

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
            stack.extend(node.inputs)

    # Start with the end node.
    # The subsequent nodes to output should only occur once it has "become"
    # a childless nodes (ie. we have already visited all of the dependencies)
    childless_nodes = [end_node]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for parent in node.inputs:
            if child_counts[parent] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent] -= 1

def numeric_gradient_check(fn, params, predictedGradients, tol=1e-6, print_progress=False):    
    numericGradients = np.zeros(len(params))
    for param in range(len(params)):
        if print_progress and param % 1000 == 0:
            print("Param {}/{}".format(param, len(params)))
        saved = params[param]

        params[param] += tol
        loss1 = fn(params)
        params[param] = saved

        params[param] -= tol
        loss2 = fn(params)
        params[param] = saved

        numericGradients[param] = (loss1 - loss2) / (2*tol)

    numericGradients = np.round(numericGradients, 12)
    predictedGradients = np.round(predictedGradients, 12)
    numericGradients += 1e-12
    predictedGradients += 1e-12
    ratio = np.linalg.norm(numericGradients - predictedGradients)
    ratio /= max(np.linalg.norm(numericGradients), np.linalg.norm(predictedGradients))
    return numericGradients, ratio





