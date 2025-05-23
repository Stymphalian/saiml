import numpy 
import cupy as cp
from typing import *
from ..base import Operator, Tensor, TensorTuple
from devices import xp
import devices


def get_broadcast_shape(x, axis):
    """
    Given the axis in which an operation was applied to x. 
    Get the new output_shape with (1,) put into the axis
    """
    if axis is None:
        outGradShape = (1,)*x.ndim
        outGradSize = x.size
    else:
        outGradShape = numpy.array(x.shape)                
        axes = list(axis) if isinstance(axis, (list, tuple)) else [axis]
        outGradSize = numpy.prod(outGradShape[axes])
        outGradShape[axes] = 1
    return tuple(outGradShape), outGradSize

def get_broadcasting_axes(a_shape, b_shape):
    axes = []
    a_ndim = len(a_shape)
    b_ndim = len(b_shape)
    if a_ndim < b_ndim:
        a_shape = (1,)*(b_ndim - a_ndim) + a_shape
        axes.extend(range(b_ndim - a_ndim, b_ndim))
    else:
        b_shape = (1,)*(a_ndim - b_ndim) + b_shape
        axes.extend(range(a_ndim - b_ndim, a_ndim))
    
    assert len(a_shape) == len(b_shape)
    num_axes = len(a_shape)

    axes = []
    for axis in range(num_axes):
        if a_shape[axis] != b_shape[axis]:
            axes.append(axis)
    return tuple(axes)

def reshape_to_match_shape(a, b):
    a_shape = a.shape
    b_shape = b.shape
    if a.ndim < b.ndim:
        a_shape = (1,) * (b.ndim - a.ndim) + a_shape
    elif a.ndim > b.ndim:
        b_shape = (1,) * (a.ndim - b.ndim) + b_shape
    return a_shape, b_shape

def make_axes_positive(axes, ndim):
    return [axis if axis >= 0 else ndim + axis for axis in axes]

class TensorAdd(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        assert len(inputs) == 2
        a = inputs[0].value()
        b = inputs[1].value()
        y = xp.add(a, b)
        assert y.shape == a.shape or y.shape == b.shape
        return y

    def gradients(self, node, outGrad):
        a = node.inputs[0]
        b = node.inputs[1]
        da = outGrad
        db = outGrad

        if a.size < da.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            da = summation(da, axis=axes).reshape(a.shape)
        if b.size < db.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            db = summation(db, axis=axes).reshape(b.shape)
        assert da.shape == a.shape
        assert db.shape == b.shape
        return (da, db)

class TensorAddScalar(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value() + self.scalar
    def gradients(self, node, outGrad):
        assert outGrad.shape == node.inputs[0].shape
        return outGrad
class TensorSub(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return xp.subtract(inputs[0].value(), inputs[1].value())
    def gradients(self, node, outGrad):
        a = node.inputs[0]
        b = node.inputs[1]
        da = outGrad
        db = -outGrad
        if a.size < da.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            da = summation(da, axis=axes).reshape(a.shape)
        if b.size < db.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            db = summation(db, axis=axes).reshape(b.shape)
        assert da.shape == a.shape
        assert db.shape == b.shape
        return (da, db)
    
class TensorSubScalar(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value() - self.scalar
    def gradients(self, node, outGrad):
        assert outGrad.shape == node.inputs[0].shape
        return outGrad
        
class TensorMult(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return xp.multiply(inputs[0].value(), inputs[1].value())
    def gradients(self, node, outGrad):
        a = node.inputs[0]
        b = node.inputs[1]
        da = outGrad * b
        db = outGrad * a

        if a.size < da.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            da = summation(da, axis=axes).reshape(a.shape)
        if b.size < db.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            db = summation(db, axis=axes).reshape(b.shape)
        assert da.shape == a.shape
        assert db.shape == b.shape

        return (da, db)
    
class TensorMultScalar(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value() * self.scalar
    def gradients(self, node, outGrad):
        dz = outGrad * self.scalar
        assert dz.shape == node.inputs[0].shape
        return dz
    
class TensorDiv(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return xp.divide(inputs[0].value(), inputs[1].value())

    def gradients(self, node, outGrad):
        a = node.inputs[0]
        b = node.inputs[1]
        da = outGrad / b
        db = -outGrad * a / (b * b)

        if a.size < da.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            da = summation(da, axis=axes).reshape(a.shape)
        if b.size < db.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            db = summation(db, axis=axes).reshape(b.shape)
        assert da.shape == a.shape
        assert db.shape == b.shape

        return (da,db)
    
class TensorDivScalar(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value() / self.scalar
    def gradients(self, node, outGrad):
        dz = outGrad / self.scalar
        assert dz.shape == node.inputs[0].shape
        return dz

class TensorSin(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return xp.sin(inputs[0].value())
    def gradients(self, node, outGrad):
        dz = outGrad * cos(node.inputs[0])
        assert dz.shape == node.inputs[0].shape
        return dz
    
class TensorCos(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return xp.cos(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        dz = outGrad * -sin(x)
        assert dz.shape == node.inputs[0].shape
        return dz
    
class TensorTan(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return xp.tan(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        dz = mult(outGrad, Tensor(1.0) / power(cos(x), 2))
        assert dz.shape == node.inputs[0].shape
        return dz
    
class TensorLog(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return xp.log(inputs[0].value())
    def gradients(self, node, outGrad):
        dz = mult(outGrad, 1.0 / node.inputs[0])
        assert dz.shape == node.inputs[0].shape
        return dz
    
class TensorMatMul(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        X = inputs[0].value()
        W = inputs[1].value()
        return xp.matmul(X, W)
    def gradients(self, node, outGrad):
        X = node.inputs[0]
        W = node.inputs[1]
        if X.ndim > 2 or W.ndim > 2:
            # x_shape, w_shape = reshape_to_match_shape(X, W)
            # X = reshape(X, x_shape)
            # W = reshape(W, w_shape)
            assert X.ndim == W.ndim
            w_axes = list(range(W.ndim-2)) + [-1,-2]
            x_axes = list(range(X.ndim-2)) + [-1,-2]
            wt = transpose(W, axis=w_axes)
            xt = transpose(X, axis=x_axes)
        else:
            xt = X.T
            wt = W.T
        dx = matmul(outGrad, wt) 
        dw = matmul(xt, outGrad) 

        x_axes = get_broadcasting_axes(dx.shape, X.shape)
        w_axes = get_broadcasting_axes(dw.shape, W.shape)
        if len(x_axes) > 0:
            dx = summation(dx, axis=x_axes).reshape(X.shape)
        if len(w_axes) > 0:
            dw = summation(dw, axis=w_axes).reshape(W.shape)
        assert dx.shape == X.shape
        assert dw.shape == W.shape
        return (dx, dw)

def parse_einsum_equation(equation:str, *operands):
    assert "->" in equation    
    in_eq, out_eq = equation.split("->")
    
    in_eqs = in_eq.strip().split(",")
    in_eqs = [eq.replace(" ", "") for eq in in_eqs]
    out_eq = out_eq.strip().replace(" ", "")

    assert len(in_eqs) == len(operands)
    for i in range(len(in_eqs)):
        assert len(in_eqs[i]) == operands[i].ndim

    return in_eqs, out_eq, operands

class TensorEinsteinSum(Operator):
    def __init__(self, equation):
        self.equation = equation

    def compute(self, *inputs: Tuple[Tensor]):
        # assert(len(inputs) == 2)
        x = [a.value() for a in inputs]
        y = xp.einsum(self.equation, *x)
        return y

    def gradients(self, node, outGrad):
        in_eqs, out_eq, ops = parse_einsum_equation(self.equation, *node.inputs)
        assert(outGrad.ndim == len(out_eq))

        derivatives = []
        for arg in range(len(in_eqs)):
            current_eq = in_eqs[arg]
            rest_eq = in_eqs[:arg] + in_eqs[arg+1:]
            current_op = ops[arg]
            rest_ops = ops[:arg] + ops[arg+1:]

            new_rest_eq = ",".join(rest_eq)
            new_rest_ops = rest_ops        
            missing = [
                (index, symbol) for index, symbol in enumerate(current_eq)
                if symbol not in out_eq
            ]
            if len(missing) > 0:
                # when the einsum output equation is missing a symbol, it means
                # we are doing a sum over that column. When computing the gradient
                # backward we need to expand out that missing dimension
                missing_eq = "".join([symbol for _, symbol in missing])
                missing_shapes = [current_op.shape[index] for index, _ in missing]
                missing_ops = Tensor(xp.ones(missing_shapes))

                new_rest_eq = ",".join([missing_eq] + rest_eq)
                new_rest_ops = (missing_ops,) + rest_ops
            
            new_out_eq = f"{out_eq},{new_rest_eq}->{current_eq}"
            dx = einsum(new_out_eq, outGrad, *new_rest_ops)
            derivatives.append(dx)

        for xi, x in enumerate(node.inputs):
            assert derivatives[xi].shape == x.shape
        return tuple(derivatives)
    
class TensorSum(Operator):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        y = xp.sum(x, axis=self.axis, keepdims=self.keepdims)
        # if self.keepdims:
        #     if self.axis is None:
        #         x_shape= (1,)*x.ndim    
        #     else:
        #         x_shape = list(x.shape)
        #         x_shape[self.axis] = 1
        #         x_shape = tuple(x_shape)
        #     assert y.shape == x_shape
        return y

    def gradients(self, node: Tensor, outGrad: Tensor):
        x = node.inputs[0]
        assert outGrad.ndim <= x.ndim
        # if outGrad.ndim < x.ndim:
        if outGrad.shape != x.shape:
            outGradShape, _ = get_broadcast_shape(x, self.axis)
            outGrad = outGrad.reshape(outGradShape)
        dz = broadcast(outGrad, x.shape)
        assert dz.shape == x.shape
        return dz
    
class TensorExp(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return xp.exp(inputs[0].value())
    def gradients(self, node, outGrad):
        x = exp(node.inputs[0])
        dx = mult(outGrad, x)
        assert dx.shape == x.shape
        return dx
    
class TensorMean(Operator):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
    def compute(self, *inputs: Tuple[Tensor]):
        return xp.mean(
            inputs[0].value(), 
            axis=self.axis,
            keepdims=self.keepdims
        )
    def gradients(self, node, outGrad):
        # axis == None: (x,y,z) -> (1,1,1)
        # axis == 1:    (x,y,z) -> (x,1,z)
        # axes == (1,2) (x,y,z) -> (x,)
        x = node.inputs[0]
        outGradSize = x.size
        if outGrad.shape != x.shape:
            outGradShape, outGradSize = get_broadcast_shape(x, self.axis)
            outGrad = outGrad.reshape(outGradShape)
        dx = broadcast(outGrad, x.shape) / outGradSize

        assert dx.shape == x.shape
        return dx
    
class TensorPower(Operator):
    def __init__(self, power, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.power = power
    def compute(self, *inputs: Tuple[Tensor]):
        if self.power < 0:
            return 1.0 / xp.power(inputs[0].value(), abs(self.power))
        else:
            return xp.power(inputs[0].value(), self.power)
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        dx = power(x, self.power - 1) * self.power
        dx = mult(outGrad, dx)
        assert (dx.shape == x.shape)
        return dx
    

argmax_axes_kernel = cp.RawKernel(
r"""

void unravel_index(int index, const int* shape, const int *dims, int ndim, int *out_coords) {
    for (int i = ndim - 1; i >= 0; --i) {
        int dim = dims[i];
        out_coords[i] = index % shape[dim];
        index /= shape[dim];
    }
}

int ravel_index(int *coords, const int* shape, const int ndim) {
    int cumulative_shape[20];
    int value = 1;
    for (int i = ndim-1; i >= 0; i--) {
        cumulative_shape[i] = value;
        value *= shape[i];
    }

    int index = 0;
    for (int i = 0; i < ndim; i++) {
        index += coords[i] * cumulative_shape[i];
    }
    return index;
}

void merge_coords(
    int* a, 
    const int *a_axis,
    const int a_ndim,
    int* b, 
    const int *b_axis,
    const int b_ndim,
    int* out,
    const int out_ndim
) {

    int ai = 0;
    int bi = 0;
    for(int i = 0; i < out_ndim; i++) {
        if (ai < a_ndim && i == a_axis[ai]){
            out[i] = a[ai++];
        } else if (bi < b_ndim) {
            out[i] = b[bi++];
        }
    }
}

void print_coords(int* coords, const int* axis, int ndim) {
    for(int i = 0; i < ndim; i++) {
        if (axis == nullptr) {
            printf("(%d:%d), ", i, coords[i]);
        } else {
            printf("(%d:%d), ", axis[i], coords[i]);
        }

    }
    printf("\n");
}

extern "C" __global__
void argmax_axes_kernel(
    const double *x,
    long long* output,
    int flat_red,         int flat_rest,
    const int* x_shape,   int n_x_shape,    
    const int* axes,      int n_axes,       
    const int* rest_axes, int n_rest_axes
) {
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    if (tId >= flat_rest) {
        return;
    }
    assert(n_x_shape <= 20);

    int rest_coords[20];
    int red_coords[20];
    int full_coords[20];

    // Get the coordinates for the rest axes given the current tId
    unravel_index(tId, x_shape, rest_axes, n_rest_axes, rest_coords);

    // Get the max
    double max_val = 0;
    int max_idx = -1;
    for(int i = 0; i < flat_red; i++) {
        unravel_index(i, x_shape, axes, n_axes, red_coords);
        merge_coords(
            rest_coords, rest_axes, n_rest_axes,
            red_coords, axes, n_axes,
            full_coords, n_x_shape
        );

        int idx = ravel_index(full_coords, x_shape, n_x_shape);
        double val = x[idx];
        if (max_idx == -1 || val > max_val) {
            max_val = val;
            max_idx = idx;    
        }
    }

    output[tId] = max_idx;
}                                                       
""", "argmax_axes_kernel")
def argmax_axes_vectorized_kernel(x: xp.ndarray, axes, keepdims=True):
    assert isinstance(x, xp.ndarray)
    axes = [a if a >= 0 else x.ndim + a for a in axes]
    axes = sorted(axes)
    all_axes = list(range(x.ndim))
    rest_axes = [a for a in all_axes if a not in axes]
    rest_shape = tuple(x.shape[a] for a in rest_axes)
    red_shape = tuple(x.shape[a] for a in axes)
    flat_rest = int(numpy.prod(rest_shape))
    flat_red = int(numpy.prod(red_shape))
    output = cp.zeros(rest_shape, dtype=cp.int64)

    num_blocks = (flat_rest // 32) + 1
    num_threads = (flat_rest // num_blocks) + 1
    # print("Flat", flat_rest, "NumBlocks: ", num_blocks, " NumThreads: ", num_threads)
    argmax_axes_kernel(
        (num_blocks,), (num_threads,), 
        (
            x, output,
            flat_red, flat_rest,
            cp.array(x.shape, dtype=cp.int32), len(x.shape),
            cp.array(axes, dtype=cp.int32), len(axes),
            cp.array(rest_axes, dtype=cp.int32), len(rest_axes)
        )
    )

    if keepdims:
        keepdims_shape = [1 for _ in range(x.ndim)]
        for axis in rest_axes:
            keepdims_shape[axis] = x.shape[axis]
        output = output.reshape(keepdims_shape)
    return output

def argmax_axes_vectorized(x: xp.ndarray, axes, keepdims=True):
    assert isinstance(x, xp.ndarray)

    axes = [a if a >= 0 else x.ndim + a for a in axes]
    axes = sorted(axes)
    all_axes = list(range(x.ndim))
    rest_axes = [a for a in all_axes if a not in axes]

    permuted_axes = rest_axes + axes
    x_perm = x.transpose(permuted_axes)

    rest_shape = tuple(x.shape[a] for a in rest_axes)
    red_shape = tuple(x.shape[a] for a in axes)

    flat_rest = int(xp.prod(xp.array(rest_shape)))
    flat_red = int(xp.prod(xp.array(red_shape)))

    x_reshaped = x_perm.reshape(flat_rest, flat_red)
    local_argmax = xp.argmax(x_reshaped, axis=1)

    if axes == all_axes[-len(axes):]:
        # if the axes are the last "right" side dimensions, we can do an optimization
        # for recovering the coordinates of the argmax
        y1 = xp.arange(flat_rest)*flat_red + local_argmax
        y1 = y1.reshape(rest_shape + (1,) * len(axes))
        # y1 = y1.transpose(np.argsort(permuted_axes))
        result = y1
    else:
        local_coords = xp.stack(xp.unravel_index(local_argmax, red_shape), axis=1)
        rest_coords = xp.stack(xp.unravel_index(xp.arange(flat_rest), rest_shape), axis=1)
        full_coords = []
        rest_index = local_index = 0
        for i in range(x.ndim):
            if i in rest_axes:
                full_coords.append(rest_coords[:, rest_index])
                rest_index += 1
            else:
                full_coords.append(local_coords[:, local_index])
                local_index += 1

        full_coords = xp.stack(full_coords, axis=0)
        flat_indices = xp.ravel_multi_index(full_coords, dims=x.shape)

        out_shape = [x.shape[a] if a not in axes else 1 for a in range(x.ndim)]
        result = flat_indices.reshape(rest_shape + (1,) * len(axes)).reshape(out_shape)

    if not keepdims:
        result = result.squeeze()
    return result
        
class TensorMax(Operator):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
    def compute(self, *inputs: Tuple[Tensor]):
        return xp.max(inputs[0].value(), axis=self.axis, keepdims=self.keepdims)
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        if self.axis is None:
            xi = xp.argmax(x.value())
            dx = xp.zeros(x.shape)
            xp.put(dx, xi, 1)
        else:
            axis = self.axis
            if isinstance(self.axis, int):
                axis = (self.axis,)
            if xp == cp:
                xi = argmax_axes_vectorized_kernel(x.value(), axis, keepdims=True)
            else:
                xi = argmax_axes_vectorized(x.value(), axis, keepdims=True)
            dx = xp.zeros(x.shape)
            xp.put(dx, xp.array(xi), 1)

            outGradShape, _ = get_broadcast_shape(x, axis)
            outGrad = outGrad.reshape(outGradShape)

        dx = outGrad * Tensor(dx)
        assert dx.shape == x.shape
        return dx
    
class TensorNegate(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return -inputs[0].value()
    def gradients(self, node, outGrad):
        assert outGrad.shape == node.inputs[0].shape
        return -outGrad
        
class TensorNorm(Operator):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
    def compute(self, *inputs: Tuple[Tensor]):
        return xp.linalg.norm(inputs[0].value(), axis=self.axis, keepdims=self.keepdims)
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        outGradShape, _ = get_broadcast_shape(x, self.axis)
        outGrad = outGrad.reshape(outGradShape)
        dx = outGrad * x / norm(x, axis=self.axis, keepdims=True)
        assert dx.shape == x.shape
        return dx
    
class TensorSqrt(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return xp.sqrt(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        dx = outGrad / (2 * sqrt(x))
        assert dx.shape == x.shape
        return dx
    
class TensorWhere(Operator):
    def __init__(self, mask):
        self.mask = mask
    def compute(self, *inputs: Tuple[Tensor]):
        a = inputs[0].value()
        b = inputs[1].value()
        z = xp.where(self.mask, a, b)
        return z
    def gradients(self, node, outGrad):
        assert outGrad.shape == node.inputs[0].shape
        a = node.inputs[0]
        b = node.inputs[1]

        assert a.shape == b.shape
        assert a.shape == outGrad.shape
        outgrad_zero = Tensor(xp.zeros(a.shape))
        da = where(self.mask, outGrad, outgrad_zero)
        db = where(self.mask, outgrad_zero, outGrad)
        return da, db
    
class TensorBitwiseOr(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        a = inputs[0].value()
        b = inputs[1].value()
        return a | b
    def gradients(self, node, outGrad):
        assert outGrad.shape == node.inputs[0].shape
        a = node.inputs[0]
        b = node.inputs[1]
        assert a.shape == b.shape
        assert a.shape == outGrad.shape
        return outGrad, outGrad
    
class TensorAbsolute(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return xp.abs(inputs[0].value())    
    def gradients(self, node, outGrad):
        assert outGrad.shape == node.inputs[0].shape
        x = node.inputs[0]
        sign = xp.sign(x.value())
        dx = outGrad * Tensor(sign)
        return dx
    
#################################################
# Tensor Tuples
#################################################

class TensorTupleMake(Operator):
    def compute(self, *inputs: Tuple[Tensor]) -> Union[xp.ndarray, Tuple[xp.ndarray]]:
        assert len(inputs) > 0
        assert isinstance(inputs[0], Tensor)
        return tuple(inputs)
    
    def gradients(self, node, outGrad):
        assert isinstance(outGrad, TensorTuple)
        dx = [x for x in outGrad]
        assert len(dx) == len(node.inputs)
        return tuple(dx)

class TensorTupleGetItem(Operator):
    def __init__(self, index):
        assert isinstance(index, int)
        self.index = index

    def compute(self, *inputs: Tuple[TensorTuple]):
        # TODO: How to handle input of tensorTuples
        assert len(inputs) == 1
        assert isinstance(inputs[0], TensorTuple)
        assert isinstance(inputs[0].value(), (list, tuple))
        assert isinstance(inputs[0].value()[self.index], Tensor)

        x_tuple = inputs[0].value()
        return x_tuple[self.index].value()

    def gradients(self, node, outGrad: Tensor):
        x = node.inputs[0]
        assert isinstance(x, TensorTuple)
        assert isinstance(outGrad, Tensor)
        
        dx = []
        for i in range(len(x)):
            if i == self.index:
                dx.append(outGrad)
            else:
                zero = Tensor(xp.zeros(x[i].shape))
                dx.append(zero)
        assert len(dx) == len(x)
        return make_tuple(*dx)
    
class TensorTupleGetSlice(Operator):
    def __init__(self, index_slice):
        assert isinstance(index_slice, slice)
        self.slice = index_slice

    def compute(self, *inputs: Tuple[TensorTuple]):
        # TODO: How to handle input of tensorTuples
        assert len(inputs) == 1
        assert isinstance(inputs[0], TensorTuple)
        assert isinstance(inputs[0].value(), (list, tuple))
        assert isinstance(inputs[0].value()[0], Tensor)

        x = inputs[0].value() # list/tuple
        return tuple(x[self.slice])

    def gradients(self, node, outGrad):
        x = node.inputs[0]
        assert isinstance(x, TensorTuple)
        assert isinstance(outGrad, TensorTuple)
        
        dx = []
        for i in range(len(x)):
            zero = Tensor(xp.zeros(x[i].shape))
            dx.append(zero)
        for i, d in enumerate(outGrad[self.slice]):
            dx[i] = d

        assert len(dx) == len(x)
        return make_tuple(*dx)

# class TensorTupleAdd(Operator):
#     def compute(self, *inputs: Tuple[TensorTuple]):
#         assert len(inputs) == 2
#         assert isinstance(inputs[0], TensorTuple)
#         assert isinstance(inputs[1], TensorTuple)
#         a = inputs[0]
#         b = inputs[1]

#         y = [a1 + b1 for a1,b1 in zip(a.value(), b.value())]
#         return tuple(y)

#     def gradients(self, node, outGrad):
#         da = outGrad
#         db = outGrad
#         return (da, db)

class TensorTupleSum(Operator):
    def compute(self, *inputs: Tuple[TensorTuple]):
        assert len(inputs) >= 1
        assert isinstance(inputs[0], TensorTuple)

        y = None
        for t in inputs:
            if y is None:
                y = list(t.value())
            else:
                for i, tensor in enumerate(t.value()):
                    y[i] += tensor
        return tuple(y)

    def gradients(self, node, outGrad: TensorTuple):
        assert isinstance(outGrad, TensorTuple)
        dz = []
        for _ in range(len(node.inputs)):
            dz.append(outGrad)
        return tuple(dz)
            
def make_tuple(*input):
    return TensorTupleMake().tensor_tuple(*input)
def tuple_get_item(x, index):
    return TensorTupleGetItem(index).tensor(x)
def tuple_get_slice(x, slice):
    return TensorTupleGetSlice(slice).tensor_tuple(x)
# def tuple_add(a, b):
#     return TensorTupleAdd().tensor_tuple(a, b)
def tuple_sum(*inputs):
    return TensorTupleSum().tensor_tuple(*inputs)

#################################################
# SHAPING OPERATIONS
#################################################

class TensorReshape(Operator):
    def __init__(self, shape):
        self.shape = shape
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value().reshape(self.shape)
    def gradients(self, node, outGrad: Tensor):
        dx = outGrad.reshape(node.inputs[0].shape)
        assert dx.shape == node.inputs[0].shape
        return dx
        
# Broadcasting allows for these cases:
# 1. Promoting '1' dimension axes: (1, 2, 1, 3) => (4, 2, 5, 3)
# 2. Left-expanding axis:          (4,) => (x, y, z, 4)
class TensorBroadcast(Operator):
    def __init__(self, shape):
        if isinstance(shape, int):
            self.shape = (shape,)
        self.shape = shape
        assert isinstance(self.shape, tuple)

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        # assert x.ndim == len(self.shape)
        return xp.broadcast_to(x, self.shape)

    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        x_shape = x.shape
        if x.ndim < outGrad.ndim:
            x_shape = (1,)*(outGrad.ndim - x.ndim) + x_shape
        assert len(x_shape) == outGrad.ndim
        # assert(x.ndim <= outGrad.ndim)

        sum_axes = []
        for axis in range(outGrad.ndim):
            s1 = x_shape[axis]
            s2 = outGrad.shape[axis]
            if s1 != s2:
                assert s1 == 1
                # The axis of the original shape is 1 so we need to do a sum 
                # along the previous axis to compute the gradient
                sum_axes.append(axis)

        dx = summation(outGrad, axis=tuple(sum_axes))
        dx = reshape(dx, x.shape)
        assert dx.shape == x.shape, f"dx shape: {dx.shape}, x shape: {x.shape}"
        return dx
    
class TensorTranspose(Operator):
    def __init__(self, axis=None):
        self.axis = axis
    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        return xp.transpose(x, self.axis)
    def gradients(self, node, outGrad):
        x = node.inputs[0]

        axes = make_axes_positive(self.axis, x.ndim)
        reverse_axes = numpy.argsort(axes)
        dx = transpose(outGrad, reverse_axes)
        assert dx.shape == x.shape
        return dx
    
class TensorRepeat(Operator):
    def __init__(self, repeats, axis=None):
        assert isinstance(repeats, int)
        self.repeats = repeats
        self.axis = axis

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        return xp.repeat(x, self.repeats, axis=self.axis)
    
    def gradients(self, node, outGrad):
        x = node.inputs[0]

        axis = self.axis
        if self.axis is None:
            axis = x.ndim - 1

        newShape = list(x.shape)
        newShape.insert(axis + 1, self.repeats)
        newShape = tuple(newShape)
        outGrad = outGrad.reshape(newShape)
        dx = summation(outGrad, axis=axis+1, keepdims=True)
        dx = dx.reshape(x.shape)
        return dx
    
class TensorTile(Operator):
    def __init__(self, repeats):
        if isinstance(repeats, int):
            repeats = (repeats,)
        self.repeats = repeats

    def _make_shape(self, x_shape, repeats):
        if len(repeats) < len(x_shape):
            repeats = (1,) * (len(x_shape) - len(repeats)) + repeats
        elif len(repeats) > len(x_shape):
            x_shape = (1,) * (len(repeats) - len(x_shape)) + x_shape
        return tuple(repeats), tuple(x_shape)

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        return xp.tile(x, self.repeats)

    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        repeats_shape, x_shape = self._make_shape(x.shape, self.repeats)
        assert len(repeats_shape) == len(x_shape)
        
        # Create the new_shape by expanding the repeated sections.
        # for example given the original shape (3,2,1) with a repeat of (4,5,6)
        # the new shape would be of shape (12, 10, 6)
        # but in order to do the sum we need to reshape the outGrad to 
        # (4,3  5,2  6,1). This allows us to do sums on the repeated axes
        new_shape = []
        for axis, r in enumerate(repeats_shape):
            new_shape.append(r)
            new_shape.append(x_shape[axis])
        new_shape = tuple(new_shape)
        sum_axes = tuple([axis * 2 for axis in range(len(repeats_shape))])

        dx = reshape(outGrad, new_shape)
        dx = summation(dx, axis=sum_axes)
        dx = reshape(dx, x.shape)
        return dx
    
class TensorStack(Operator):
    def __init__(self, axis=0):
        self.axis = axis

    def compute(self, *inputs: Tuple[TensorTuple]) -> Tensor:
        assert len(inputs) == 1
        assert isinstance(inputs[0], TensorTuple)
        ref_shape = inputs[0][0].shape
        for x in inputs[0]:
            assert x.shape == ref_shape

        flat = [x.value() for x in inputs[0]]
        y = xp.stack(flat, axis=self.axis)
        return y

    def gradients(self, node, outGrad: Tensor):
        assert isinstance(outGrad, Tensor)
        assert isinstance(node.inputs[0], TensorTuple)
        x = node.inputs[0]
        dx = unstack(outGrad, axis=self.axis)

        assert len(dx) == len(x)
        return dx
    
class TensorUnstack(Operator):
    def __init__(self, axis=0):
        self.axis = axis

    def compute(self, *inputs: Tuple[Tensor]):
        assert isinstance(inputs[0], Tensor)

        num_splits = inputs[0].shape[self.axis]
        y = xp.split(inputs[0].value(), num_splits, axis=self.axis)
        y = [xp.squeeze(a, axis=self.axis) for a in y]

        return tuple(Tensor(a) for a in y)

    def gradients(self, node, outGrad):
        dx = stack(outGrad, axis=self.axis)
        return dx
        
class TensorConcatenate(Operator):
    def __init__(self, axis=None):
        self.axis = axis

    def compute(self, *inputs: Tuple[TensorTuple]) -> xp.ndarray:
        a = [x.value() for x in inputs[0]]
        y = xp.concatenate(a, axis=self.axis)
        return y

    def gradients(self, node, outGrad:Tensor) -> TensorTuple:
        assert isinstance(outGrad, Tensor)
        x = node.inputs[0]
        if self.axis is None:
            x_shape = x[0].shape
            dx = split(outGrad, len(x), axis=0)
            dx = [reshape(a, x_shape) for a in dx]
            assert len(dx) == len(x)
            return make_tuple(*dx)            
        else:
            dx = split(outGrad, len(x), axis=self.axis)
            assert len(dx) == len(x)
            return dx
    
class TensorSplit(Operator):
    def __init__(self, num_splits, axis=0):
        self.num_splits = num_splits
        self.axis = axis

    def compute(self, *inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        x = inputs[0].value()
        y = xp.split(x, self.num_splits, axis=self.axis)
        assert len(y) == self.num_splits
        return tuple([Tensor(a) for a in y])

    def gradients(self, node, outGrad):
        assert isinstance(outGrad, TensorTuple)
        assert len(outGrad) == self.num_splits
        x = node.inputs[0].value()

        out = [a for a in outGrad]
        dx = stack(out, axis=self.axis)
        dx = reshape(dx, x.shape)

        assert dx.shape == x.shape, f"dx shape: {dx.shape}, x shape: {x.shape}"
        return dx

def constant(a):
    return Tensor(a, requires_grad=False)
def add(a, b):
    return TensorAdd().tensor(a,b)
def add_scalar(a, b):
    return TensorAddScalar(b).tensor(a)
def sub(a, b):
    return TensorSub().tensor(a, b)
def sub_scalar(a, b):
    return TensorSubScalar(b).tensor(a)
def mult(a, b):
    return TensorMult().tensor(a, b)
def mult_scalar(a, b):
    return TensorMultScalar(b).tensor(a)
def div(a, b):
    return TensorDiv().tensor(a, b)
def div_scalar(a, b):
    return TensorDivScalar(b).tensor(a)
def matmul(a, b):
    return TensorMatMul().tensor(a, b)
def einsum(equation, *inputs):
    return TensorEinsteinSum(equation).tensor(*inputs)
def sin(a):
    return TensorSin().tensor(a)
def cos(a):
    return TensorCos().tensor(a)
def tan(a):
    return TensorTan().tensor(a)
def log(a):
    return TensorLog().tensor(a)
def summation(a, axis=None, keepdims=False):
    return TensorSum(axis, keepdims).tensor(a)
def exp(a):
    return TensorExp().tensor(a)
def mean(a, axis=None, keepdims=False):
    return TensorMean(axis=axis, keepdims=keepdims).tensor(a)
def power(a, power):
    return TensorPower(power).tensor(a)
def max(a, axis=None, keepdims=False):
    return TensorMax(axis=axis, keepdims=keepdims).tensor(a)
def neg(a):
    return TensorNegate().tensor(a)
def norm(a, axis=None, keepdims=False):
    return TensorNorm(axis=axis, keepdims=keepdims).tensor(a)
def sqrt(a):
    return TensorSqrt().tensor(a)
def where(mask, a, b):
    return TensorWhere(mask).tensor(a, b)
def bitwise_or(a, b):
    return TensorBitwiseOr().tensor(a, b)
def abs(a):
    return TensorAbsolute().tensor(a)

def reshape(a, shape):
    return TensorReshape(shape).tensor(a)
def broadcast(a, shape):
    return TensorBroadcast(shape).tensor(a)
def transpose(x, axis=None):
    return TensorTranspose(axis).tensor(x)
def repeat(x, repeats, axis=None):
    return TensorRepeat(repeats, axis=axis).tensor(x)
def tile(x, repeats):
    return TensorTile(repeats).tensor(x)
def stack(inputs: List[Tensor], axis=0):
    return TensorStack(axis=axis).tensor(make_tuple(*inputs))
def unstack(a, axis=0):
    return TensorUnstack(axis=axis).tensor_tuple(a)
def concatenate(inputs: Sequence[Tensor], axis=None):
    return TensorConcatenate(axis).tensor(make_tuple(*inputs))
def split(a, num_splits, axis=0):
    return TensorSplit(num_splits, axis).tensor_tuple(a)
def vstack(arr: Sequence[Tensor]):
    return concatenate(arr, axis=0)
def vsplit(a, num_splits):
    return split(a, num_splits, axis=0)