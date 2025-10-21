import os
import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable
from typing import Any
import numpy as np

def check_inf_nan(value):
    if math.isinf(value) or math.isnan(value):
        return 0
    return value

# Function copied from verl/trainer/ppo/metric_utils.py.
def process_validation_metrics(
    data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        # if var2vals.get("pred", None) is not None:
                        #     vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                        #     [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                        #         data=vote_data,
                        #         subset_size=n,
                        #         reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                        #         seed=seed,
                        #     )
                        #     metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val

# Function copied from verl/trainer/ppo/metric_utils.py.
def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]

def print_metrics(data_src2var2metric2val):
    for data_source, var2metric2val in data_src2var2metric2val.items():
        print("-" * 40)
        for var_name, metric2val in var2metric2val.items():
            print(f"  Metric: {var_name}")
            for metric_name, metric_val in metric2val.items():
                print(f"    {metric_name}: {metric_val:.4f}")
        print("\n" + "=" * 40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    print(f"Analyzing results in {args.path}")

    jsons = os.listdir(os.path.join(args.path, 'log'))
    print(f"Found {len(jsons)} json files.")
    succ = 0
    spl = 0
    distance_to_goal = 0
    path_length = 0
    oracle_succ = 0
    
    infos_dict = defaultdict(list)
    sample_inputs = []
    data_sources = []
    
    for j in jsons:
        with open(os.path.join(args.path, 'log', j)) as f:
            try:
                data = json.load(f)
                episode_id = data['id']
                
                data_sources.append("dummy_source")  # Placeholder
                sample_inputs.append(str(episode_id))
                
                success = check_inf_nan(int(data['success']))
                infos_dict['success'].append(success)
                spl = check_inf_nan(data['spl'])
                infos_dict['spl'].append(spl)
                distance_to_goal = check_inf_nan(data['distance_to_goal'])
                infos_dict['distance_to_goal'].append(distance_to_goal)
                oracle_succ = check_inf_nan(int(data['oracle_success']))
                infos_dict['oracle_success'].append(oracle_succ)
                path_length = data['path_length']
                infos_dict['path_length'].append(path_length)
            except Exception as e:
                import traceback
                print(f"Error reading {j}, skipping. Exception: {e!r}")
                traceback.print_exc()
                continue    
    
    data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, infos_dict)
    
    print_metrics(data_src2var2metric2val)
