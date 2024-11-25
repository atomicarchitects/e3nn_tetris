# Copied from https://github.com/e3nn/e3nn-jax/blob/245e17eb23deaccad9f2c9cfd40fe40515e3c074/examples/tetris_point.py#L13

import time

import flax
import jax
import jax.numpy as jnp
import jraph
import optax
from tqdm.auto import tqdm

import e3nn_jax as e3nn
import cuequivariance_jax as cuex
import cuequivariance as cue
import cuequivariance.segmented_tensor_product as stp



from typing import Sequence, Optional
import itertools

def full_tensor_product(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3_filter: Optional[Sequence[cue.Irrep]] = None,
) -> cue.EquivariantTensorProduct:
    """
    subscripts: ``lhs[iu],rhs[jv],output[kuv]``

    Construct a weightless channelwise tensor product descriptor.

    .. currentmodule:: cuequivariance

    Args:
        irreps1 (Irreps): Irreps of the first operand.
        irreps2 (Irreps): Irreps of the second operand.
        irreps3_filter (sequence of Irrep, optional): Irreps of the output to consider.

    Returns:
        EquivariantTensorProduct: Descriptor of the full tensor product.
    """
    G = irreps1.irrep_class

    if irreps3_filter is not None:
        irreps3_filter = into_list_of_irrep(G, irreps3_filter)

    d = stp.SegmentedTensorProduct.from_subscripts("iu,jv,kuv+ijk")

    for mul, ir in irreps1:
        d.add_segment(0, (ir.dim, mul))
    for mul, ir in irreps2:
        d.add_segment(1, (ir.dim, mul))

    irreps3 = []

    for (i1, (mul1, ir1)), (i2, (mul2, ir2)) in itertools.product(
        enumerate(irreps1), enumerate(irreps2)
    ):
        for ir3 in ir1 * ir2:
            # for loop over the different solutions of the Clebsch-Gordan decomposition
            for cg in cue.clebsch_gordan(ir1, ir2, ir3):
                d.add_path(i1, i2, None, c=cg)

                irreps3.append((mul1 * mul2, ir3))

    irreps3 = cue.Irreps(G, irreps3)
    irreps3, perm, inv = irreps3.sort()
    d = d.permute_segments(2, inv)

    d = d.normalize_paths_for_operand(-1)
    return cue.EquivariantTensorProduct(
        d,
        [irreps1, irreps2, irreps3],
        layout=cue.ir_mul,
    )


def tetris() -> jraph.GraphsTuple:
    pos = [
        [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0]],  # chiral_shape_1
        [[1, 1, 1], [1, 1, 2], [2, 1, 1], [2, 0, 1]],  # chiral_shape_2
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],  # square
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],  # line
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],  # corner
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]],  # L
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]],  # T
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]],  # zigzag
    ]
    pos = jnp.array(pos, dtype=jnp.float32)
    labels = jnp.arange(8)

    graphs = []

    for p, l in zip(pos, labels):
        senders, receivers = e3nn.radius_graph(p, 1.1)

        graphs += [
            jraph.GraphsTuple(
                nodes=p.reshape((4, 3)),  # [num_nodes, 3]
                edges=None,
                globals=l[None],  # [num_graphs]
                senders=senders,  # [num_edges]
                receivers=receivers,  # [num_edges]
                n_node=jnp.array([len(p)]),  # [num_graphs]
                n_edge=jnp.array([len(senders)]),  # [num_graphs]
            )
        ]

    return jraph.batch(graphs)


class Layer(flax.linen.Module):
    target_irreps: cue.Irreps
    denominator: float
    sh_lmax: int = 3

    @flax.linen.compact
    def __call__(self, graphs, positions):
        target_irreps = cue.Irreps("O3", self.target_irreps)

        def update_edge_fn(edge_features, sender_features, receiver_features, globals):
            sh = cuex.spherical_harmonics(list(range(1, self.sh_lmax + 1)), positions[graphs.receivers] - positions[graphs.senders], True)
            tp = cuex.equivariant_tensor_product(full_tensor_product(sender_features.irreps(), sh.irreps()))(sender_features, sh)
            return cuex.concatenate(
                [sender_features, tp]
            ).regroup()

        def update_node_fn(node_features, sender_features, receiver_features, globals):
            node_feats = receiver_features / self.denominator
            node_feats = e3nn.flax.Linear(target_irreps, name="linear_pre")(node_feats)
            shortcut = e3nn.flax.Linear(
                node_feats.irreps, name="shortcut", force_irreps_out=True
            )(node_features)
            return shortcut + node_feats

        return jraph.GraphNetwork(update_edge_fn, update_node_fn)(graphs)


class Model(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, graphs):
        positions = cuex.IrrepsArray(cue.Irreps("O3", "1o"), graphs.nodes, layout=cue.ir_mul)
        graphs = graphs._replace(nodes=cuex.IrrepsArray(cue.Irreps("O3", "0e"), jnp.ones((positions.shape[0], 1)), layout=cue.ir_mul))

        layers = 2 * ["32x0e + 32x0o + 8x1e + 8x1o + 8x2e + 8x2o"] + ["0o + 7x0e"]

        for irreps in layers:
            graphs = Layer(irreps, 1.5)(graphs, positions)

        # Readout logits
        pred = e3nn.scatter_sum(
            graphs.nodes.array, nel=graphs.n_node
        )  # [num_graphs, 1 + 7]
        odd, even1, even2 = pred[:, :1], pred[:, 1:2], pred[:, 2:]
        logits = jnp.concatenate([odd * even1, -odd * even1, even2], axis=1)
        assert logits.shape == (len(graphs.n_node), 8)  # [num_graphs, num_classes]

        return logits


def train(steps=200):
    model = Model()

    # Optimizer
    opt = optax.adam(learning_rate=0.01)

    def loss_fn(params, graphs):
        logits = model.apply(params, graphs)
        labels = graphs.globals  # [num_graphs]

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = jnp.mean(loss)
        return loss, logits

    @jax.jit
    def update_fn(params, opt_state, graphs):
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, logits = grad_fn(params, graphs)
        labels = graphs.globals
        accuracy = jnp.mean(jnp.argmax(logits, axis=1) == labels)

        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, accuracy

    # Dataset
    graphs = tetris()

    # Init
    init = jax.jit(model.init)
    params = init(jax.random.PRNGKey(3), graphs)
    opt_state = opt.init(params)

    # compile jit
    wall = time.perf_counter()
    print("compiling...", flush=True)
    _, _, accuracy = update_fn(params, opt_state, graphs)
    print(f"initial accuracy = {100 * accuracy:.0f}%", flush=True)
    print(f"compilation took {time.perf_counter() - wall:.1f}s")

    # Train
    wall = time.perf_counter()
    print("training...", flush=True)
    for _ in tqdm(range(steps)):
        params, opt_state, accuracy = update_fn(params, opt_state, graphs)

        if accuracy == 1.0:
            break

    print(f"final accuracy = {100 * accuracy:.0f}%")
    print(f"training took {time.perf_counter() - wall:.1f}s")


if __name__ == "__main__":
    train()