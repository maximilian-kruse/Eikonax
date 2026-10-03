r"""Components for computing derivatives of the Eikonax solver.

This module contains two main components. Firstly, the
[`PartialDerivator`][eikonax.derivator.PartialDerivator] evaluates the partial derivatives of the
global Eikonax update operator $\mathbf{G}$ w.r.t. the parameter tensor field $\mathbf{M}$ and the
corresponding solution vector $\mathbf{u}$ obtained from a forward solve. The
[`DerivativeSolver`][eikonax.derivator.DerivativeSolver] component makes use of the fixed
point/adjoint property of the Eikonax solver to evaluate total parametric derivatives.

Classes:
    PartialDerivatorData: Settings for initialization of partial derivator
    PartialDerivator: Component for computing partial derivatives of the Godunov Update operator
    DerivativeSolver: Main component for obtaining gradients from partial derivatives
"""

from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from beartype.vale import Is
from jaxtyping import Float as jtFloat
from jaxtyping import Int as jtInt

from . import corefunctions, linalg, preprocessing


@dataclass
class PartialDerivatorData:
    """Settings for initialization of partial derivator.

    See the [Forward Solver](../usage/solve.md) documentation for more detailed explanations.

    Attributes:
        softminmax_order (int): Order of the the soft minmax function for differentiable
            transformation of the update parameters
        softminmax_cutoff (Real): Cut-off in for minmax transformation, beyond which zero
            sensitivity is assumed.
    """

    use_soft_update: bool
    softminmax_order: Annotated[int, Is[lambda x: x > 0]]
    softminmax_cutoff: Annotated[Real, Is[lambda x: x > 0]]


# ==================================================================================================
class PartialDerivator(eqx.Module):
    r"""Component for computing partial derivatives of the Godunov Update operator.

    Given a tensor field $M$ and a solution vector $u$, the partial derivator computes the partial
    derivatives of the global Eikonax update operator with respect to the solution vector,
    $\mathbf{G}_u(\mathbf{u}, \mathbf{m})$, and the tensor field,
    $\mathbf{G}_M(\mathbf{u}, \mathbf{M})$. All derivatives are computed on the vertex level,
    exploiting the locality of interactions in the update operator (only adjacent simplices are
    considered). Therefore, we can indeed assemble the complete derivative operators as parse data
    structures, not just Jacobian-vector or vector-Jacobian products, within a single pass over the
    computational mesh. Atomic functions on the vertex level are differentiated with Jax.

    !!! info
        For the computation of the derivatives, so-called 'self-updates' are disabled. These updates
        occur when a vertex does not receive a lower update value from any direction than the value
        it currently has. At the correct solution point, this case cannot occur due to the causality
        of the update stencil.

    Methods:
        compute_partial_derivatives: Compute the partial derivatives of the Godunov update operator
            with respect to the solution vector and the parameter tensor field, given a state for
            both variables
    """

    # Equinox modules are data classes, so we need to specify attributes on class level
    _num_vertices: int
    _num_simplices: int
    _vertices: jtFloat[jax.Array, "num_vertices dim"]
    _adjacency_data: jtInt[jax.Array, "num_vertices max_num_adjacent_simplices 4"]
    _initial_site_inds: jtInt[jax.Array, "num_initial_sites"]
    _use_soft_update: bool
    _softminmax_order: int
    _softminmax_cutoff: int

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        mesh_data: preprocessing.MeshData,
        derivator_data: PartialDerivatorData,
        initial_sites: preprocessing.InitialSites,
    ) -> None:
        """Constructor for the partial derivator object.

        Args:
            mesh_data (preprocessing.MeshData): Mesh data object also utilized for the Eikonax
                solver, contains adjacency data for every vertex.
            derivator_data (PartialDerivatorData): Settings for initialization of the derivator.
            initial_sites (preprocessing.InitialSites): Locations and values at source points
        """
        self._num_vertices = mesh_data.num_vertices
        self._num_simplices = mesh_data.num_simplices
        self._vertices = mesh_data.vertices
        self._adjacency_data = mesh_data.adjacency_data
        self._initial_site_inds = initial_sites.inds
        self._use_soft_update = derivator_data.use_soft_update
        self._softminmax_order = derivator_data.softminmax_order
        self._softminmax_cutoff = derivator_data.softminmax_cutoff

    # ----------------------------------------------------------------------------------------------
    def compute_partial_derivatives(
        self,
        solution_vector: jtFloat[jax.Array | npt.NDArray, "num_vertices"],
        tensor_field: jtFloat[jax.Array | npt.NDArray, "num_simplices dim dim"],
    ) -> tuple[linalg.EikonaxSparseMatrix, linalg.DerivatorSparseTensor]:
        r"""Compute the partial derivatives of the Godunov update operator.

        This method provides the main interface for computing the partial derivatives of the global
        Eikonax update operator with respect to the solution vector and the parameter tensor field.
        The updates are computed locally for each vertex, such that the resulting data structures
        are sparse. Subsequently, further zero entries are removed to reduce the memory footprint.
        The derivatives computed in this component can be utilized to compute the total parametric
        derivative via a fix point equation, given that the provided solution vector is that
        fix point.
        The computation of partial derivatives is possible with a single pass over the mesh, since
        the solution of the Eikonax equation, and therefore causality within the Godunov update
        scheme, is known.

        !!! note
            The derivator expects the metric tensor field as used in the inner product for the
            update stencil of the eikonal equation. This is the **INVERSE** of the conductivity
            tensor, which is the actual tensor field in the eikonal equation. The
            [`Tensorfield`][eikonax.tensorfield.TensorField] component provides the inverse tensor
            field.

        Args:
            solution_vector (jax.Array): Current solution
            tensor_field (jax.Array): Parameter field

        Returns:
            tuple[eikonax.linalg.EikonaxSparseMatrix, eikonax.linalg.DerivatorSparseTensor]:
                The partial derivatives G_u and G_M packaged in the project's sparse data
                structures. [`EikonaxSparseMatrix`][eikonax.linalg.EikonaxSparseMatrix] contains the
                (row_inds, col_inds, values, shape) representation used by the project and can be
                converted to a SciPy sparse array via the
                [`convert_to_scipy_sparse`][eikonax.linalg.convert_to_scipy_sparse] method.
                G_u corresponds to a square operator of shape (N_V, N_V). G_M is returned as a
                [`DerivatorSparseTensor`][eikonax.linalg.DerivatorSparseTensor] holding the
                per-vertex derivative values with shape (N_V, N_S_adj, d, d) and index
                information for adjacent simplices.
        """
        solution_vector = jnp.array(solution_vector, dtype=jnp.float32)
        tensor_field = jnp.array(tensor_field, dtype=jnp.float32)
        partial_derivative_solution, partial_derivative_parameter = (
            self._compute_global_partial_derivatives(
                solution_vector,
                tensor_field,
            )
        )
        partial_derivative_solution = self._process_partial_derivative_solution(
            partial_derivative_solution
        )
        partial_derivative_parameter = self._process_partial_derivative_parameter(
            partial_derivative_parameter
        )
        return partial_derivative_solution, partial_derivative_parameter

    # ----------------------------------------------------------------------------------------------
    def _process_partial_derivative_solution(
        self,
        partial_derivative_solution: jtFloat[
            jax.Array, "num_vertices max_num_adjacent_simplices 2"
        ],
    ) -> linalg.EikonaxSparseMatrix:
        """Convert dense partial derivatives into the project's sparse matrix representation.

        This method transforms the dense vertex-wise derivative array into an
        [`EikonaxSparseMatrix`][eikonax.linalg.EikonaxSparseMatrix] by extracting row indices,
        column indices, and values. Derivatives at initial/boundary sites are zeroed out since
        these vertices have fixed values.

        Args:
            partial_derivative_solution (jax.Array): Dense array of shape
                (num_vertices, max_num_adjacent_simplices, 2). The last axis contains the
                derivative contributions for the two neighboring vertices participating in each
                local Godunov update stencil.

        Returns:
            eikonax.linalg.EikonaxSparseMatrix: Sparse container representing the partial
                derivative matrix G_u with shape (num_vertices, num_vertices). Can be converted
                to a SciPy sparse array via
                [`convert_to_scipy_sparse`][eikonax.linalg.convert_to_scipy_sparse].
        """
        partial_derivative_solution = partial_derivative_solution.at[
            self._initial_site_inds, ...
        ].set(0.0)
        max_num_adjacent_vertices = self._adjacency_data.shape[1]
        row_inds = jnp.repeat(jnp.arange(self._num_vertices), 2 * max_num_adjacent_vertices)
        col_inds = self._adjacency_data[..., 1:3].flatten()
        values = partial_derivative_solution.flatten()
        eikonax_sparse_matrix = linalg.EikonaxSparseMatrix(
            row_inds=row_inds,
            col_inds=col_inds,
            values=values,
            shape=(self._num_vertices, self._num_vertices),
        )
        return eikonax_sparse_matrix

    # ----------------------------------------------------------------------------------------------
    def _process_partial_derivative_parameter(
        self,
        partial_derivative_parameter: jtFloat[
            jax.Array, "num_vertices max_num_adjacent_simplices dim dim"
        ],
    ) -> linalg.DerivatorSparseTensor:
        """Package the parameter derivative tensor into DerivatorSparseTensor.

        This method wraps the dense tensor of parameter derivatives (with respect to the metric
        tensor field) into the project's sparse tensor data structure, which pairs the derivative
        values with simplex adjacency information.

        Args:
            partial_derivative_parameter (jax.Array): Dense array of shape
                (num_vertices, max_num_adjacent_simplices, dim, dim). Contains the partial
                derivatives of the update operator with respect to the metric tensor components
                of each adjacent simplex.

        Returns:
            eikonax.linalg.DerivatorSparseTensor: Sparse tensor wrapper holding derivative values
                and simplex connectivity metadata from ``self._adjacency_data``.
        """
        partial_derivative_parameter = partial_derivative_parameter.at[
            self._initial_site_inds, ...
        ].set(0.0)
        derivator_sparse_tensor = linalg.DerivatorSparseTensor(
            derivative_values=partial_derivative_parameter,
            adjacent_simplex_data=self._adjacency_data[..., 3],
        )
        return derivator_sparse_tensor

    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def _compute_global_partial_derivatives(
        self,
        solution_vector: jtFloat[jax.Array, "num_vertices"],
        tensor_field: jtFloat[jax.Array, "num_simplices dim dim"],
    ) -> tuple[
        jtFloat[jax.Array, "num_vertices max_num_adjacent_simplices 2"],
        jtFloat[jax.Array, "num_vertices max_num_adjacent_simplices dim dim"],
    ]:
        """Compute partial derivatives of the global update operator.

        The method is a jitted and vectorized call to the
        [`_compute_vertex_partial_derivative`][eikonax.derivator.PartialDerivator._compute_vertex_partial_derivatives]
        method.

        Args:
            solution_vector (jax.Array): Global solution vector
            tensor_field (jax.Array): Global parameter tensor field

        Returns:
            tuple[jax.Array, jax.Array]: Raw data for partial derivatives, with shapes
                (N, num_adjacent_simplices, 2) and (N, num_adjacent_simplices, dim, dim)
        """
        global_partial_derivative_function = jax.vmap(
            self._compute_vertex_partial_derivatives,
            in_axes=(None, None, 0),
        )
        partial_derivative_solution, partial_derivative_parameter = (
            global_partial_derivative_function(solution_vector, tensor_field, self._adjacency_data)
        )
        return partial_derivative_solution, partial_derivative_parameter

    # ----------------------------------------------------------------------------------------------
    def _compute_vertex_partial_derivatives(
        self,
        solution_vector: jtFloat[jax.Array, "num_vertices"],
        tensor_field: jtFloat[jax.Array, "num_simplices dim dim"],
        adjacency_data: jtInt[jax.Array, "max_num_adjacent_simplices 4"],
    ) -> tuple[
        jtFloat[jax.Array, "max_num_adjacent_simplices 2"],
        jtFloat[jax.Array, "max_num_adjacent_simplices dim dim"],
    ]:
        """Compute partial derivatives for the update of a single vertex.

        The method computes candidates for all respective subterms through calls to further methods.
        These candidates are filtered for feasibility by means of JAX filters.
        The softmin function (and its gradient) is applied to the directions of all optimal
        updates to ensure differentiability, other contributions are discarded.
        Lastly, the evaluated contributions are combined according to the form of the
        "total differential" for the partial derivatives.

        Args:
            solution_vector (jax.Array): Global solution vector
            tensor_field (jax.Array): Global parameter tensor field
            adjacency_data (jax.Array): Adjacency data for the vertex under consideration

        Returns:
            tuple[jax.Array, jax.Array]: Partial derivatives for the given vertex
        """
        max_num_adjacent_simplices = adjacency_data.shape[0]
        tensor_dim = tensor_field.shape[1]
        assert adjacency_data.shape == (max_num_adjacent_simplices, 4), (
            f"node-level adjacency data needs to have shape ({max_num_adjacent_simplices}, 4), "
            f"but has shape {adjacency_data.shape}"
        )

        vertex_update_candidates = corefunctions.compute_vertex_update_candidates(
            solution_vector,
            tensor_field,
            adjacency_data,
            self._vertices,
            self._use_soft_update,
            self._softminmax_order,
            self._softminmax_cutoff,
        )
        grad_update_solution_candidates, grad_update_parameter_candidates = (
            self._compute_vertex_partial_derivative_candidates(
                solution_vector, tensor_field, adjacency_data
            )
        )
        min_value, grad_update_solution_candidates, grad_update_parameter_candidates = (
            self._filter_candidates(
                vertex_update_candidates,
                grad_update_solution_candidates,
                grad_update_parameter_candidates,
            )
        )
        average_grad = corefunctions.grad_average(
            vertex_update_candidates.flatten(), min_value
        ).reshape(vertex_update_candidates.shape)

        grad_update_solution = jnp.zeros((max_num_adjacent_simplices, 2))
        grad_update_parameter = jnp.zeros((max_num_adjacent_simplices, tensor_dim, tensor_dim))
        for i in range(max_num_adjacent_simplices):
            grad_update_solution = grad_update_solution.at[i, :].set(
                jnp.tensordot(average_grad[i, :], grad_update_solution_candidates[i, ...], axes=1)
            )
            grad_update_parameter = grad_update_parameter.at[i, ...].set(
                jnp.tensordot(average_grad[i, :], grad_update_parameter_candidates[i, ...], axes=1)
            )

        return grad_update_solution, grad_update_parameter

    # ----------------------------------------------------------------------------------------------
    def _compute_vertex_partial_derivative_candidates(
        self,
        solution_vector: jtFloat[jax.Array, "num_vertices"],
        tensor_field: jtFloat[jax.Array, "num_simplices dim dim"],
        adjacency_data: jtInt[jax.Array, "max_num_adjacent_simplices 4"],
    ) -> tuple[
        jtFloat[jax.Array, "max_num_adjacent_simplices 4 2"],
        jtFloat[jax.Array, "max_num_adjacent_simplices 4 dim dim"],
    ]:
        """Compute partial derivatives corresponding to potential update candidates for a vertex.

        Update candidates and corresponding derivatives are computed for all adjacent simplices,
        and for all possible update parameters per simplex.

        Args:
            solution_vector (jax.Array): Global solution vector
            tensor_field (jax.Array): Global parameter field
            adjacency_data (jax.Array): Adjacency data for the given vertex

        Returns:
            tuple[jax.Array, jax.Array]: Candidates for partial derivatives
        """
        max_num_adjacent_simplices = adjacency_data.shape[0]
        tensor_dim = tensor_field.shape[1]
        grad_update_solution_candidates = jnp.zeros(
            (max_num_adjacent_simplices, 4, 2), dtype=jnp.float32
        )
        grad_update_parameter_candidates = jnp.zeros(
            (max_num_adjacent_simplices, 4, tensor_dim, tensor_dim), dtype=jnp.float32
        )

        for i, indices in enumerate(adjacency_data):
            partial_solution, partial_parameter = (
                self._compute_partial_derivative_candidates_from_adjacent_simplex(
                    solution_vector, tensor_field, indices
                )
            )
            grad_update_solution_candidates = grad_update_solution_candidates.at[i, ...].set(
                partial_solution
            )
            grad_update_parameter_candidates = grad_update_parameter_candidates.at[i, ...].set(
                partial_parameter
            )

        return grad_update_solution_candidates, grad_update_parameter_candidates

    # ----------------------------------------------------------------------------------------------
    def _compute_partial_derivative_candidates_from_adjacent_simplex(
        self,
        solution_vector: jtFloat[jax.Array, "num_vertices"],
        tensor_field: jtFloat[jax.Array, "num_simplices dim dim"],
        adjacency_data: jtInt[jax.Array, "4"],
    ) -> tuple[jtFloat[jax.Array, "4 2"], jtFloat[jax.Array, "4 dim dim"]]:
        r"""Compute partial derivatives for all update candidates within an adjacent simplex.

        The update candidates are evaluated according to the different candidates for the
        optimization parameters $\lambda$. Contributions are combined to the form of the involved
        total differentials.

        Args:
            solution_vector (jax.Array): Global solution vector
            tensor_field (jax.Array): Flobal parameter field
            adjacency_data (jax.Array): Adjacency data for the given vertex and simplex

        Returns:
            tuple[jax.Array, jax.Array]: Derivative candidate from the given simplex
        """
        i, j, k, s = adjacency_data
        tensor_dim = tensor_field.shape[1]
        solution_values = jnp.array((solution_vector[j], solution_vector[k]))
        edges = corefunctions.compute_edges(i, j, k, self._vertices)
        parameter_tensor = tensor_field[s]
        if self._use_soft_update:
            lambda_array = corefunctions.compute_optimal_update_parameters_soft(
                solution_values,
                parameter_tensor,
                edges,
                self._softminmax_order,
                self._softminmax_cutoff,
            )
        else:
            lambda_array = corefunctions.compute_optimal_update_parameters_hard(
                solution_values, parameter_tensor, edges
            )
        lambda_partial_solution, lambda_partial_parameter = self._compute_lambda_grad(
            solution_values, parameter_tensor, edges
        )
        lambda_partial_solution = jnp.concatenate((jnp.zeros((2, 2)), lambda_partial_solution))
        lambda_partial_parameter = jnp.concatenate(
            (jnp.zeros((2, tensor_dim, tensor_dim)), lambda_partial_parameter)
        )
        grad_update_solution = jnp.zeros((4, 2))
        grad_update_parameter = jnp.zeros((4, tensor_dim, tensor_dim))

        for i in range(4):
            update_partial_lambda = corefunctions.grad_update_lambda(
                solution_values, parameter_tensor, lambda_array[i], edges
            )
            update_partial_solution = corefunctions.grad_update_solution(
                solution_values, parameter_tensor, lambda_array[i], edges
            )
            update_partial_parameter = corefunctions.grad_update_parameter(
                solution_values, parameter_tensor, lambda_array[i], edges
            )
            grad_update_solution = grad_update_solution.at[i, :].set(
                update_partial_lambda * lambda_partial_solution[i, :] + update_partial_solution
            )
            grad_update_parameter = grad_update_parameter.at[i, ...].set(
                update_partial_lambda * lambda_partial_parameter[i, ...] + update_partial_parameter
            )
        return grad_update_solution, grad_update_parameter

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _filter_candidates(
        vertex_update_candidates: jtFloat[jax.Array, "max_num_adjacent_simplices 4"],
        grad_update_solution_candidates: jtFloat[jax.Array, "max_num_adjacent_simplices 4 2"],
        grad_update_parameter_candidates: jtFloat[
            jax.Array, "max_num_adjacent_simplices 4 dim dim"
        ],
    ) -> tuple[
        jtFloat[jax.Array, ""],
        jtFloat[jax.Array, "max_num_adjacent_simplices 4 2"],
        jtFloat[jax.Array, "max_num_adjacent_simplices 4 dim dim"],
    ]:
        """Mask irrelevant derivative candidates so that they are discarded later.

        Values are masked by setting them to zero or infinity, depending on the routine in which
        they are utilized later. Partial derivatives are only relevant if the corresponding update
        corresponds to an optimal path.

        Args:
            vertex_update_candidates (jax.Array): Update candidates for a given vertex
            grad_update_solution_candidates (jax.Array): Partial derivative candidates w.r.t. the
                solution vector
            grad_update_parameter_candidates (jax.Array): Partial derivative candidates w.r.t. the
                parameter field

        Returns:
            tuple[jax.Array, jax.Array, jax.Array]: Optimal update value, masked partial
                derivatives
        """
        min_value = jnp.min(vertex_update_candidates)
        min_candidate_mask = jnp.where(vertex_update_candidates == min_value, 1, 0)
        vertex_update_candidates = jnp.where(
            min_candidate_mask == 1, vertex_update_candidates, jnp.inf
        )
        grad_update_solution_candidates = jnp.where(
            min_candidate_mask[..., None] == 1, grad_update_solution_candidates, 0
        )
        grad_update_parameter_candidates = jnp.where(
            min_candidate_mask[..., None, None] == 1, grad_update_parameter_candidates, 0
        )

        return min_value, grad_update_solution_candidates, grad_update_parameter_candidates

    # ----------------------------------------------------------------------------------------------
    def _compute_lambda_grad(
        self,
        solution_values: jtFloat[jax.Array, "2"],
        parameter_tensor: jtFloat[jax.Array, "dim dim"],
        edges: tuple[
            jtFloat[jax.Array, "dim"], jtFloat[jax.Array, "dim"], jtFloat[jax.Array, "dim"]
        ],
    ) -> tuple[jtFloat[jax.Array, "4 2"], jtFloat[jax.Array, "4 dim dim"]]:
        """Compute the partial derivatives of update parameters for a single vertex.

        This method evaluates the partial derivatives of the update parameters with respect to the
        current solution vector and the given parameter field, for a single triangle.

        Args:
            solution_values (jax.Array): Current solution values at the opposite vertices of the
                considered triangle
            parameter_tensor (jax.Array): Parameter tensor for the given triangle
            edges (tuple[jax.Array, jax.Array, jax.Array]): Edges of the considered triangle

        Returns:
            tuple[jax.Array, jax.Array]: Jacobians of the update parameters w.r.t.
                the solution vector and the parameter tensor
        """
        if self._use_soft_update:
            lambda_partial_solution = corefunctions.jac_lambda_soft_solution(
                solution_values,
                parameter_tensor,
                edges,
                self._softminmax_order,
                self._softminmax_cutoff,
            )
            lambda_partial_parameter = corefunctions.jac_lambda_soft_parameter(
                solution_values,
                parameter_tensor,
                edges,
                self._softminmax_order,
                self._softminmax_cutoff,
            )
        else:
            lambda_partial_solution = corefunctions.jac_lambda_hard_solution(
                solution_values, parameter_tensor, edges
            )
            lambda_partial_parameter = corefunctions.jac_lambda_hard_parameter(
                solution_values, parameter_tensor, edges
            )

        return lambda_partial_solution, lambda_partial_parameter


# ==================================================================================================
class DerivativeSolver:
    r"""Main component for obtaining gradients from partial derivatives.

    The Eikonax [`PartialDerivator`][eikonax.derivator.PartialDerivator] computes partial
    derivatives of the global update operator with respect
    to the solution vector, $\mathbf{G}_u$, and the parameter tensor field, $\mathbf{G}_M$.
    We exploit the fact that the obtained solution candidate from a forward solve
    $\mathbf{u}\in\mathbb{R}^{N_V}$ is, up to a given accuracy, a fixed point of the
    global update operator. We further consider the scenario of $\mathbf{M}(\mathbf{m})$ being
    dependent on some parameter $\mathbf{m}\in\mathbb{R}^M$. This means we can write $\mathbf{u}$ as
    a function of $\mathbf{m}$, obeying the relation

    $$
        \mathbf{u}(\mathbf{m}) = \mathbf{G}(\mathbf{u}(\mathbf{m}), \mathbf{M}(\mathbf{m}))
    $$

    To obtain the Jacobian
    $\mathbf{J} = \frac{d\mathbf{u}}{d\mathbf{m}}\in\mathbb{R}^{N_V\times M}$,
    we simply differentiate the fixed point relation,

    $$
        \mathbf{J} = \mathbf{G}_u\mathbf{J}
        + \overbrace{\mathbf{G}_M\frac{d\mathbf{M}(\mathbf{m})}{d\mathbf{m}}}^{\mathbf{G}_m}
        \quad\Leftrightarrow\quad \mathbf{J} = (\mathbf{I} - \mathbf{G}_u)^{-1}\mathbf{G}_m
    $$

    $\mathbf{G}_u$ and $\mathbf{G}_M$ are provided by the
    [`PartialDerivator`][eikonax.derivator.PartialDerivator], whereas
    $\frac{d\mathbf{M}}{d\mathbf{m}}$ is computed as the Jacobian of the
    [`TensorField`][eikonax.tensorfield.TensorField] component.

    We are typically not interested in the full Jacobian, but rather in the gradient of some
    cost functional $l:\mathbb{R}^{N_V}\to\mathbb{R},\ l=l(\mathbf{u}(\mathbf{m}))$ with respect to
    $\mathbf{m}$. The gradient is given as

    $$
        \mathbf{g}(\mathbf{m}) = \frac{d l}{d\mathbf{m}}
        = \overbrace{\frac{d l}{d\mathbf{u}}}^{l_u^T}\mathbf{J}
        = \overbrace{l_u^T(\mathbf{I} - \mathbf{G}_u)^{-1}}^{\mathbf{v}^T}\mathbf{G}_m.
    $$

    We can identify $\mathbf{v}$ as the adjoint variable, which is obtained by solving the linear
    **discrete adjoint equation**,

    $$
        (\mathbf{I} - \mathbf{G}_u)^T\mathbf{v} = l_u.
    $$

    *Now comes the catch*: Through the strict causality of the Godunov update operator, we can find
    a unique and consistent ordering of vertex indices, such that the solution at a vertex $i$ is
    only informed by the solution at a vertex $j$, if $j$ occurs before $i$ in that ordering. The
    matrix $\mathbf{G}_u$ has to form a directed, acyclic graph.
    This means that there is an orthogonal permutation matrix $\mathbf{P}$ such that for
    $\bar{\mathbf{G}}_u = \mathbf{P}\mathbf{G}_u\mathbf{P}^T$ an entry $(\bar{\mathbf{G}}_u)_{ij}$
    is only non-zero if $i > j$. In total, we can write

    $$
    \begin{align*}
    (\mathbf{I}-\mathbf{G}_u)^T\mathbf{v} = \mathbf{l}_u &\Leftrightarrow
    \mathbf{P}(\mathbf{I}-\mathbf{G}_u)^T\mathbf{P}^T
    \overbrace{\mathbf{P}\mathbf{v}}^{\bar{\mathbf{v}}}
    = \overbrace{\mathbf{P}\mathbf{l}_u}^{\bar{\mathbf{l}}_u} \nonumber \\
    & \Leftrightarrow \overbrace{(\mathbf{I} - \bar{\mathbf{G}}_u)^T}^{\bar{\mathbf{A}}}\bar{v}
    =\bar{\mathbf{l}}_u
    \end{align*}
    $$

    where $\bar{\mathbf{A}}$ is an upper triangular matrix with unit diagonal. Hence, it is
    invertible through simple back-substitution.

    The `DerivativeSolver` component does exactly this: It sets up the matrices $\mathbf{P}$ and
    $\bar{\mathbf{A}}$, permutates inputs/outputs, and solves the sparse linear system through
    back-substitution.

    !!! tip "Speedy gradients"
        Given a solution vector $\mathbf{u}$, Eikonax computes derivatives with linear complexity.
        Even more, for a given evaluation point, we can evaluate an arbitrary number of gradients
        through simple backsubstitution. All matrices need to be assembled only once.

    !!! info "Change in tooling"
        In the `DerivativeSolver`, we leave JAX and fall back to the `numpy`/`scipy` stack. While
        the sequential solver operation should not be mush slower on the CPU, we have to transfer
        the data back from the offloading device. We plan to implement a GPU-compatible solver with
        [CuPy](https://cupy.dev/) in a future version, or in JAX as soon as it offers the necessary
        linear algebra tools.

    Methods:
        solve: Solve the linear system for the adjoint variable
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        solution: jtFloat[jax.Array | npt.NDArray, "num_vertices"],
        sparse_partial_update_solution: sp.sparray,
    ) -> None:
        r"""Constructor for the derivative solver.

        Initializes the causality-inspired permutation matrix $\mathbf{P}$, and afterwards the
        permuted system matrix $\bar{\mathbf{A}}$, which is triangular.

        Args:
            solution (jax.Array | npt.NDArray): Obtained solution of the Eikonal equation.
            sparse_partial_update_solution (scipy.sparray): Sparse matrix
                representing $\mathbf{G}_u$. Any SciPy sparse format is accepted as long
                as it supports conversion to CSC via ``.tocsc()`` (e.g. ``coo_array``,
                ``coo_matrix``, ``csr_matrix``). The implementation will convert it to
                CSC internally.
        """
        num_points = solution.size
        solution = np.array(solution, dtype=np.float32)
        self._sparse_permutation_matrix = self._assemble_permutation_matrix(solution)
        self._sparse_system_matrix = self._assemble_system_matrix(
            sparse_partial_update_solution, num_points
        )

    # ----------------------------------------------------------------------------------------------
    def solve(
        self, right_hand_side: jtFloat[jax.Array | npt.NDArray, "num_vertices"]
    ) -> jtFloat[npt.NDArray, "num_parameters"]:
        r"""Solve the linear system for the parametric gradient.

        Following the notation from the class docstring,this method solves the linear system for
        the adjoint variable $\mathbf{v}$. Given a right-hand-side $\mathbf{l}_u$, this is a three-
        step process:

        1. Permute the right hand side $\bar{l}_u = \mathbf{P}l_u$
        2. Solve the linear system $\bar{\mathbf{A}}\bar{\mathbf{v}} = \bar{l}_u$
        3. Permute solution back to the original ordering
            $\mathbf{v} = \mathbf{P}^T\bar{\mathbf{v}}$

        Args:
            right_hand_side (jax.Array | npt.NDArray): RHS for the linear system solve

        Returns:
            np.ndarray: Solution of the linear system solve, corresponding to the adjoint in an
                optimization context.
        """
        right_hand_side = np.array(right_hand_side, dtype=np.float32)
        permutated_right_hand_side = self._sparse_permutation_matrix @ right_hand_side
        permutated_solution = sp.linalg.spsolve_triangular(
            self._sparse_system_matrix, permutated_right_hand_side, lower=False, unit_diagonal=True
        )
        solution = self._sparse_permutation_matrix.T @ permutated_solution

        return solution

    # ----------------------------------------------------------------------------------------------
    def _assemble_permutation_matrix(
        self, solution: jtFloat[npt.NDArray, "num_vertices"]
    ) -> sp.csc_matrix:
        r"""Construct permutation matrix $\mathbf{P}$ for index ordering."""
        num_points = solution.size
        permutation_row_inds = np.arange(solution.size)
        permutation_col_inds = np.argsort(solution)
        permutation_values = np.ones(solution.size)
        sparse_permutation_matrix = sp.csc_matrix(
            (permutation_values, (permutation_row_inds, permutation_col_inds)),
            shape=(num_points, num_points),
        )

        return sparse_permutation_matrix

    # ----------------------------------------------------------------------------------------------
    def _assemble_system_matrix(
        self,
        sparse_partial_update_solution: sp.sparray,
        num_points: int,
    ) -> sp.csc_matrix:
        r"""Assemble system matrix $\bar{\mathbf{A}}$ for gradient solver.

        Before invoking this method, the permutation matrix $\mathbf{P}$ must be initialized.

        Args:
            sparse_partial_update_solution (scipy.sparse matrix-like): Sparse matrix
                representing $\mathbf{G}_u$. Will be converted to CSC internally using
                ``.tocsc()``.
            num_points (int): Number of points (vertices) in the system; used to build
                the identity matrix and shape the final system matrix.
        """
        sparse_partial_update_solution = sparse_partial_update_solution.tocsc()
        sparse_identity_matrix = sp.identity(num_points, format="csc")
        sparse_system_matrix = sparse_identity_matrix - sparse_partial_update_solution
        sparse_system_matrix = (
            self._sparse_permutation_matrix
            @ sparse_system_matrix.T
            @ self._sparse_permutation_matrix.T
        )

        return sparse_system_matrix

    # ----------------------------------------------------------------------------------------------
    @property
    def sparse_system_matrix(self) -> sp.csc_matrix:
        r"""Get system matrix $\bar{\mathbf{A}}\in\mathbb{R}^{{N_V}\times {N_V}}$."""
        return self._sparse_system_matrix

    # ----------------------------------------------------------------------------------------------
    @property
    def sparse_permutation_matrix(self) -> sp.csc_matrix:
        r"""Get permutation matrix $\mathbf{P}\in\mathbb{R}^{{N_V}\times {N_V}}$."""
        return self._sparse_permutation_matrix


# ==================================================================================================
def compute_eikonax_jacobian(
    derivative_solver: DerivativeSolver,
    partial_derivative_parameter: sp.sparray,
) -> jtFloat[npt.NDArray, "num_vertices num_parameters"]:
    """Compute Jacobian from concatenation of gradients, computed with unit vector RHS.

    !!! warning

        This method should only be used for small problems.

    Args:
        derivative_solver (DerivativeSolver): Initialized derivative solver object.
        partial_derivative_parameter (scipy.sparse matrix-like or numpy.ndarray):
            Partial derivative of the global update operator with respect to the parameter
            vector/parameters. Should support matrix-vector multiplication or transposed
            multiplication with the adjoint vector; typical inputs are SciPy sparse
            matrices/arrays (e.g. ``coo_array``, ``csr_matrix``) or dense NumPy arrays with
            shape (N_V, M).

    Returns:
        jtFloat[npt.NDArray, "num_vertices num_parameters"]: Dense Jacobian matrix with shape
            (N_V, M). Note: this routine builds the full dense Jacobian and is intended only for
            small problems.
    """
    rhs_adjoint = np.zeros(derivative_solver.sparse_permutation_matrix.shape[0])
    jacobian = []

    for i, _ in enumerate(rhs_adjoint):
        rhs_adjoint[i] = 1.0
        adjoint = derivative_solver.solve(rhs_adjoint)
        rhs_adjoint[i] = 0.0
        jacobian_row = partial_derivative_parameter.T @ adjoint
        jacobian.append(jacobian_row)
    jacobian = np.vstack(jacobian)
    return jacobian


# --------------------------------------------------------------------------------------------------
def compute_eikonax_hessian() -> None:
    """Compute Hessian matrix.

    !!! failure "Not implemented yet"
    """
    raise NotImplementedError
