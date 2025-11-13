import string
from itertools import chain, combinations
from typing import List

import numpy as np
import torch

EPSILON = np.finfo(np.float32).tiny


class ConfigDict(object):
    def __init__(self, dict) -> None:
        self.dict = dict
        for k, v in dict.items():
            setattr(self, k, v)

    def get(self, key):
        return self.dict.get(key)


# ---- ground truth content style retrievel for numerical simulation -----------
# ------------------------------------------------------------------------------
def powerset(iterable, only_consider_whole_set=False):
    """
    Generate all subsets of views with at least two elements.

    Args:
        iterable: An iterable object containing the views.
        only_consider_whole_set: A boolean indicating whether to consider only the whole subset.

    Returns:
        A tuple containing two lists:
        - The first list contains all subsets of views with at least two elements.
        - The second list contains binary indicators showing whether a specific view is included in each subset.
    """
    s = list(iterable)
    sets = list(
        chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1))
    )
    if only_consider_whole_set:
        ps_leq_2 = [
            s for s in sets if len(s) == len(list(iterable))
        ]  # if consider the whole subset
    else:
        ps_leq_2 = [s for s in sets if len(s) > 1]
    binary_indicator = [[int(view in s) for view in iterable] for s in ps_leq_2]
    return ps_leq_2, binary_indicator


def retrieve_content_style(zs):
    """
    Retrieve the content and style components from a list of zs.

    Parameters:
    zs (list): List of zs where each zs represents the latent space of a view.

    Returns:
    tuple: A tuple containing the content and style components.
           The content component is a set of common elements across all views.
           The style component is a list of sets, where each set represents the unique elements for each view.
    """
    zs = zs.tolist()
    # zs: shape: [n_views * nz]
    content = set(zs[0])
    for i in range(1, len(zs)):
        content.intersection_update(set(zs[i]))
    style = [set(z_Sk).difference(content) for z_Sk in zs]
    return content, style


def content_style_from_subsets(subsets, zs):
    """
    Retrieve content and style from subsets of zs.

    Args:
        subsets (list): List of subsets.
        zs (numpy.ndarray): Array of zs.

    Returns:
        tuple: A tuple containing two dictionaries - content_dict and style_dict.
            - content_dict: A dictionary mapping each subset to its corresponding content.
            - style_dict: A dictionary mapping each subset to its corresponding style.

    """
    content_dict, style_dict = {}, {}
    for subset in subsets:
        content, style = retrieve_content_style(zs[subset, :])
        if len(content) == 0:
            continue
        else:
            content_dict[subset] = content
            style_dict[subset] = {k: style[i] for i, k in enumerate(subset)}
    return content_dict, style_dict


def unpack_item_list(lst):
    if isinstance(lst, tuple):
        lst = list(lst)
    result_list = []
    for it in lst:
        if isinstance(it, (tuple, list)):
            result_list.append(unpack_item_list(it))
        else:
            result_list.append(it.item())
    return result_list


# ----------- exp-name generator --------------
# ---------------------------------------------


def valid_str(v):
    if hasattr(v, "__name__"):
        return valid_str(v.__name__)
    if isinstance(v, tuple) or isinstance(v, list):
        return "-".join([valid_str(x) for x in v])
    str_v = str(v).lower()
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    str_v = "".join(c if c in valid_chars else "-" for c in str_v)
    return str_v


def get_exp_name(
    args,
    parser,
    blacklist=[
        "evaluate",
        "num_train_batches",
        "num_eval_batches",
        "evaluate_iter",
    ],
):
    exp_name = ""
    for x in vars(args):
        if getattr(args, x) != parser.get_default(x) and x not in blacklist:
            if isinstance(getattr(args, x), bool):
                exp_name += ("_" + x) if getattr(args, x) else ""
            else:
                exp_name += "_" + x + valid_str(getattr(args, x))
    return exp_name.lstrip("_")


# ----------- content mask-related utils  ----
# ---------------------------------------------
def topk_gumbel_softmax(k, logits, tau, hard=True):
    """
    Applies the top-k Gumbel-Softmax operation to the input logits.

    Args:
        k (int): The number of elements to select from the logits.
        logits (torch.Tensor): The input logits.
        tau (float): The temperature parameter for the Gumbel-Softmax operation.
        hard (bool, optional): Whether to use the straight-through approximation.
            If True, the output will be a one-hot vector. If False, the output will be a
            continuous approximation of the top-k elements. Default is True.

    Returns:
        torch.Tensor: The output tensor after applying the top-k Gumbel-Softmax operation.
    """
    m = torch.distributions.gumbel.Gumbel(
        torch.zeros_like(logits), torch.ones_like(logits)
    )
    g = m.sample()
    logits = logits + g

    # continuous top k
    khot = torch.zeros_like(logits).type_as(logits)
    onehot_approx = torch.zeros_like(logits).type_as(logits)
    for i in range(k):
        khot_mask = torch.max(
            1.0 - onehot_approx, torch.tensor([EPSILON]).type_as(logits)
        )
        logits = logits + torch.log(khot_mask)
        onehot_approx = torch.nn.functional.softmax(logits / tau, dim=1)
        khot = khot + onehot_approx

    if hard:
        # straight through
        khot_hard = torch.zeros_like(khot)
        val, ind = torch.topk(khot, k, dim=1)
        khot_hard = khot_hard.scatter_(1, ind, 1)
        res = khot_hard - khot.detach() + khot
    else:
        res = khot
    return res


def mask2indices(masks, keys):
    """
    Convert binary masks to indices of non-zero elements for each key.

    Args:
        masks (list): List of binary masks.
        keys (list): List of keys corresponding to the masks.

    Returns:
        dict: Dictionary mapping each key to a list of indices of non-zero elements in the corresponding mask.
    """
    estimated_content_indices = {}
    assert len(keys) == len(masks)
    for k, c_mask in zip(keys, masks):
        c_ind = torch.where(c_mask)[-1].tolist()
        estimated_content_indices[k] = c_ind
    return estimated_content_indices


def gumbel_softmax_mask(
    avg_logits: torch.Tensor, subsets: List, content_sizes: List
):
    """
    Applies the Gumbel-Softmax function to generate masks for each subset.

    Args:
        avg_logits (torch.Tensor): The average logits for each subset.
        subsets (List): The list of subsets.
        conten_sizes (List): The list of content sizes for each subset.

    Returns:
        List: The list of masks generated using Gumbel-Softmax for each subset.
    """
    masks = []
    for i, subset in enumerate(subsets):
        m = topk_gumbel_softmax(
            k=content_sizes[i], logits=avg_logits, tau=1.0, hard=True
        )
        masks += [m]
    return masks


def smart_gumbel_softmax_mask(
    avg_logits: torch.Tensor, subsets: List, content_sizes: List
):
    """
    Generates masks using smart Gumbel softmax for each subset.

    Args:
        avg_logits (torch.Tensor): Average logits.
        subsets (List): List of subsets.
        conten_sizes (List): List of content sizes.

    Returns:
        List: List of masks for each subset.
    """
    masks = []
    joint_content_size = content_sizes[-1]
    joint_content_mask = torch.eye(avg_logits.shape[-1])[:2].type_as(avg_logits)

    for i, subset in enumerate(subsets[:-1]):
        m = topk_gumbel_softmax(
            k=content_sizes[i] - joint_content_size,
            logits=avg_logits,
            tau=1.0,
            hard=True,
        )
        m = torch.concat([joint_content_mask, m], 0)
        masks += [m]
    return masks


# ----------- Evaluation-related utils ------------
# -------------------------------------------------
def evaluate_prediction(model, metric, X_train, y_train, X_test, y_test):
    """
    Evaluates the performance of a model by fitting it on training data, predicting on test data,
    and calculating a specified metric between the predicted and true labels.

    Parameters:
        model (object): The machine learning model to be evaluated.
        metric (function): The evaluation metric to be used.
        X_train (array-like): The training input samples.
        y_train (array-like): The training target values.
        X_test (array-like): The test input samples.
        y_test (array-like): The test target values.

    Returns:
        float: The evaluation score calculated using the specified metric.
    """
    # handle edge cases when inputs or labels are zero-dimensional
    if any([0 in x.shape for x in [X_train, y_train, X_test, y_test]]):
        return np.nan
    assert X_train.shape[1] == X_test.shape[1]
    if y_train.ndim > 1:
        assert y_train.shape[1] == y_test.shape[1]
        if y_train.shape[1] == 1:
            y_train = y_train.ravel()
            y_test = y_test.ravel()
    # handle edge cases when the inputs are one-dimensional
    if X_train.shape[1] == 1:
        X_train = X_train.reshape(-1, 1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metric(y_test, y_pred)


def generate_batch_factor_code(
    ground_truth_data,
    representation_function,
    num_points,
    random_state,
    batch_size,
):
    """Sample a single training sample based on a mini-batch of ground-truth data.

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observation as input and
        outputs a representation.
      num_points: Number of points to sample.
      random_state: Numpy random state used for randomness.
      batch_size: Batchsize to sample points.

    Returns:
      representations: Codes (num_codes, num_points)-np array.
      factors: Factors generating the codes (num_factors, num_points)-np array.
    """
    representations = None
    factors = None
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_factors, current_observations = ground_truth_data.sample(
            num_points_iter, random_state
        )
        if i == 0:
            factors = current_factors
            representations = representation_function(current_observations)
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack(
                (representations, representation_function(current_observations))
            )
        i += num_points_iter
    return np.transpose(representations), np.transpose(factors)


# ----------- Utilities from Causal Component Analysis ------------
# https://github.com/akekic/causal-component-analysis/blob/main/data_generator/utils.py

from pathlib import Path
from typing import Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.distributions import Uniform


def leaky_tanh(x: Tensor, alpha: float = 1.0, beta: float = 0.1) -> Tensor:
    return torch.tanh(alpha * x) + beta * x


def leaky_relu(x: Tensor, alpha: float = 0.5) -> Tensor:
    return torch.nn.functional.leaky_relu(x, negative_slope=alpha)


def leaky_sigmoid(x: Tensor, alpha: float = 1.0, beta: float = 0.1) -> Tensor:
    return torch.sigmoid(alpha * x) + beta * x


def summary_statistics(
    x: Tensor, v: Tensor, e: Tensor, intervention_targets: Tensor
) -> dict[str, pd.DataFrame]:
    x_summary_stats = pd.DataFrame(x.numpy()).describe().T.rename_axis("index")
    v_summary_stats = pd.DataFrame(v.numpy()).describe().T.rename_axis("index")
    e_summary_stats = pd.DataFrame(e.numpy()).describe().T.rename_axis("index")
    intervention_targets_summary_stats = (
        pd.DataFrame(intervention_targets.numpy()).describe().T
    ).rename_axis("index")
    return {
        "x": x_summary_stats,
        "v": v_summary_stats,
        "e": e_summary_stats,
        "intervention_targets_per_env": intervention_targets_summary_stats,
    }


def plot_dag(adj_matrix: np.ndarray, log_dir: Path) -> None:
    G = nx.DiGraph(adj_matrix)

    fig = plt.figure()
    for layer, nodes in enumerate(nx.topological_generations(G)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            G.nodes[node]["layer"] = layer
    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos, with_labels=True, width=4.0, node_size=700, arrowsize=20)
    plt.savefig(log_dir / "dag.png")
    # release memory
    fig.clf()
    plt.close("all")


def random_perm(num_causal_variables: int) -> torch.Tensor:
    while True:
        perm = torch.randperm(num_causal_variables)
        if torch.all(perm != torch.arange(num_causal_variables)):
            return perm


def sample_invertible_matrix(n: int) -> Tensor:
    matrix = torch.rand((n, n))
    while torch.abs(torch.det(matrix)) < 0.1:
        matrix = torch.randn((n, n))
    return matrix


def sample_coeffs(
    low: float = 0.0,
    high: float = 1.0,
    size: tuple[int] = (1,),
    min_abs_value: Optional[float] = None,
) -> Tensor:
    if min_abs_value is not None:
        assert min_abs_value < max(abs(low), abs(high))
        while True:
            coeffs = Uniform(low, high).sample(size)
            if torch.all(torch.abs(coeffs) >= min_abs_value):
                return coeffs
    else:
        return Uniform(low, high).sample(size)


def linear_base_func(
    v: Tensor, u: Tensor, index: int, parents: Tensor, coeffs: Tensor
) -> Tensor:
    assert len(parents) + 1 == len(coeffs)

    if len(parents) == 0:
        return coeffs * u[:, index]
    else:
        vec = torch.concatenate(
            (v[:, parents].T, u[:, index].unsqueeze(0)), dim=0
        )
        return coeffs.matmul(vec)


def linear_inverse_jacobian(
    v: Tensor, u: Tensor, index: int, parents: Tensor, coeffs: Tensor
) -> Tensor:
    assert len(parents) + 1 == len(coeffs)

    if len(parents) == 0:
        return torch.ones_like(u[:, index])
    else:
        return torch.abs(coeffs[-1] * torch.ones_like(u[:, index]))


def sample_random_matrix(*size: int) -> Tensor:
    return torch.randn(*size)


def make_random_nonlinear_func(
    input_dim: int, output_dim: int, n_nonlinearities: int
) -> callable:
    assert input_dim > 0, "input_dim must be positive"
    assert output_dim > 0, "output_dim must be positive"
    assert n_nonlinearities > 0, "must have at least one nonlinearity"

    matrices = []
    for i in range(n_nonlinearities - 1):
        matrices.append(sample_random_matrix(input_dim, input_dim))
    matrices.append(sample_random_matrix(output_dim, input_dim))

    nonlinearities = []
    for i in range(n_nonlinearities):
        nonlinearities.append(leaky_tanh)

    def nonlinear_func(input: Tensor) -> Tensor:
        output = input.T
        for i in range(n_nonlinearities):
            output = nonlinearities[i](matrices[i].matmul(output))
        return output.T

    return nonlinear_func


def make_location_scale_function(
    index: int,
    parents: Union[list[int], Tensor],
    n_nonlinearities: int,
    snr: float = 1.0,
) -> tuple[callable, callable]:
    if len(parents) == 0:
        return lambda v, u: u[:, index], lambda v, u: torch.ones_like(
            u[:, index]
        )

    loc_func = make_random_nonlinear_func(len(parents), 1, n_nonlinearities)
    scale_func = make_random_nonlinear_func(len(parents), 1, n_nonlinearities)

    def location_scale_func(v, u):
        loc = loc_func(v[:, parents])
        scale = scale_func(v[:, parents])
        return (snr * loc + scale * u[:, index].unsqueeze(1)).squeeze(1)

    def inverse_jacobian(v, u):
        return torch.abs(1.0 / scale_func(v[:, parents])).squeeze(1)

    return location_scale_func, inverse_jacobian
