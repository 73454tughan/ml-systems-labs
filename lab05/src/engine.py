"""
Scalar Automatic Differentiation Engine
Author: Tughan Bulut Kurtulus (Student ID: 73454)
Course: Machine Learning Systems (Lab 05)
Institution: Vistula University

Implements a scalar DAG computation graph node with dynamic reverse-mode automatic differentiation.
"""

from __future__ import annotations
import math
from typing import Set, Tuple, Optional, Callable, Union


class ScalarNode:
    """
    A scalar Value node representing a node in a dynamic computation graph.
    Maintains the forward scalar value, accumulated gradient, input ancestors,
    and a local gradient backward closure.
    """
    __slots__ = ('val', 'grad', '_creators', '_backward_fn', '_operation', 'tag')

    def __init__(
        self,
        val: Union[int, float],
        _children: Tuple[ScalarNode, ...] = (),
        _op: str = '',
        tag: Optional[str] = None
    ) -> None:
        self.val: float = float(val)
        self.grad: float = 0.0
        self._creators: Set[ScalarNode] = set(_children)
        self._backward_fn: Callable[[], None] = lambda: None
        self._operation: str = _op
        self.tag: Optional[str] = tag

    def __repr__(self) -> str:
        tag_str = f", tag='{self.tag}'" if self.tag else ""
        return f"ScalarNode(val={self.val:.6f}, grad={self.grad:.6f}{tag_str})"

    def __add__(self, other: Union[ScalarNode, int, float]) -> ScalarNode:
        other_node = other if isinstance(other, ScalarNode) else ScalarNode(other)
        out = ScalarNode(self.val + other_node.val, (self, other_node), '+')

        def _backward() -> None:
            # d(x + y)/dx = 1, d(x + y)/dy = 1
            self.grad += 1.0 * out.grad
            other_node.grad += 1.0 * out.grad

        out._backward_fn = _backward
        return out

    def __radd__(self, other: Union[int, float]) -> ScalarNode:
        return self.__add__(other)

    def __mul__(self, other: Union[ScalarNode, int, float]) -> ScalarNode:
        other_node = other if isinstance(other, ScalarNode) else ScalarNode(other)
        out = ScalarNode(self.val * other_node.val, (self, other_node), '*')

        def _backward() -> None:
            # d(x * y)/dx = y, d(x * y)/dy = x
            self.grad += other_node.val * out.grad
            other_node.grad += self.val * out.grad

        out._backward_fn = _backward
        return out

    def __rmul__(self, other: Union[int, float]) -> ScalarNode:
        return self.__mul__(other)

    def __pow__(self, exponent: Union[int, float]) -> ScalarNode:
        assert isinstance(exponent, (int, float)), "Power exponent must be scalar int or float"
        out = ScalarNode(self.val ** exponent, (self,), f'**{exponent}')

        def _backward() -> None:
            # d(x^p)/dx = p * x^(p-1)
            self.grad += (exponent * (self.val ** (exponent - 1))) * out.grad

        out._backward_fn = _backward
        return out

    def relu(self) -> ScalarNode:
        out_val = self.val if self.val > 0.0 else 0.0
        out = ScalarNode(out_val, (self,), 'ReLU')

        def _backward() -> None:
            # d(ReLU(x))/dx = 1 if x > 0 else 0
            self.grad += (1.0 if self.val > 0.0 else 0.0) * out.grad

        out._backward_fn = _backward
        return out

    def sigmoid(self) -> ScalarNode:
        # Numerically stable sigmoid evaluation
        s = 1.0 / (1.0 + math.exp(-self.val)) if self.val >= 0 else math.exp(self.val) / (1.0 + math.exp(self.val))
        out = ScalarNode(s, (self,), 'Sigmoid')

        def _backward() -> None:
            # d(sigma(x))/dx = sigma(x) * (1 - sigma(x))
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward_fn = _backward
        return out

    def log(self) -> ScalarNode:
        # Clamped logarithm to prevent log(0) domain error
        clamped_val = max(self.val, 1e-12)
        out = ScalarNode(math.log(clamped_val), (self,), 'Log')

        def _backward() -> None:
            # d(ln(x))/dx = 1 / x
            self.grad += (1.0 / clamped_val) * out.grad

        out._backward_fn = _backward
        return out

    def __neg__(self) -> ScalarNode:
        return self * -1.0

    def __sub__(self, other: Union[ScalarNode, int, float]) -> ScalarNode:
        return self + (-other)

    def __rsub__(self, other: Union[int, float]) -> ScalarNode:
        return ScalarNode(other) + (-self)

    def __truediv__(self, other: Union[ScalarNode, int, float]) -> ScalarNode:
        other_node = other if isinstance(other, ScalarNode) else ScalarNode(other)
        return self * (other_node ** -1.0)

    def backward(self) -> None:
        """
        Executes reverse-mode automatic differentiation:
        1. Topologically sorts the computation graph DAG via depth-first search.
        2. Sets root gradient to 1.0.
        3. Propagates adjoints backwards in reverse topological order.
        4. Accumulates gradients to support multi-consumer subgraphs.
        """
        ordered_nodes = []
        visited = set()

        def build_topo(v: ScalarNode) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._creators:
                    build_topo(child)
                ordered_nodes.append(v)

        build_topo(self)

        # Seed the output gradient
        self.grad = 1.0

        # Traverse in reverse topological sequence
        for node in reversed(ordered_nodes):
            node._backward_fn()
