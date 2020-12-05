"""
Collect metrics with low memory requirements (avoid collecting superfluous data)
"""
from typing import Any
from ray.util.iter import LocalIterator
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.metric_ops import OncePerTimeInterval
from ray.tune.integration.wandb import wandb_mixin


def LowMemoryMetricsReporting(train_op: LocalIterator[Any], workers: WorkerSet,
                             config: dict) -> LocalIterator[dict]:
    output_op = train_op \
        .filter(OncePerTimeInterval(max(2, config["min_iter_time_s"]))) \
        .for_each(CollectMetricsLowMem(
            workers, min_history=config["metrics_smoothing_episodes"],
            timeout_seconds=config["collect_metrics_timeout"]))
    return output_op


class CollectMetricsLowMem:
    def __init__(self, workers, min_history=100, timeout_seconds=180, log_to_neptune=False):
        self.workers = workers
        self.episode_history = []
        self.to_be_collected = []
        self.min_history = min_history
        self.timeout_seconds = timeout_seconds
        self.log_to_neptune = log_to_neptune

    def __call__(self, _):
        # Collect worker metrics.
        episodes, self.to_be_collected = collect_episodes(
            self.workers.local_worker(),
            self.workers.remote_workers(),
            self.to_be_collected,
            timeout_seconds=self.timeout_seconds)
        orig_episodes = list(episodes)
        missing = self.min_history - len(episodes)
        if missing > 0:
            episodes.extend(self.episode_history[-missing:])
            assert len(episodes) <= self.min_history
        self.episode_history.extend(orig_episodes)
        self.episode_history = self.episode_history[-self.min_history:]
        res = summarize_episodes(episodes, orig_episodes)

        # Add in iterator metrics.
        metrics = LocalIterator.get_metrics()
        timers = {}
        counters = {}
        info = {}
        info.update(metrics.info)
        for k, counter in metrics.counters.items():
            counters[k] = counter
        for k, timer in metrics.timers.items():
            timers["{}_time_ms".format(k)] = round(timer.mean * 1000, 3)
            if timer.has_units_processed():
                timers["{}_throughput".format(k)] = round(
                    timer.mean_throughput, 3)
        res.update({
            "num_healthy_workers": len(self.workers.remote_workers()),
            "timesteps_total": metrics.counters[STEPS_SAMPLED_COUNTER],
        })
        res["timers"] = timers
        res["info"] = info
        res["info"].update(counters)
        relevant = ["info", "custom_metrics", "sampler_perf", "timesteps_total", "policy_reward_mean",
                    "episode_len_mean"]

        d = {
            k: res[k] for k in relevant
        }
        d["evaluation"] = res.get("evaluation", {})

        if self.log_to_neptune:
            metrics_to_be_logged = ["info", "evaluation"]
            def log_metric(metrics, base_string=''):
                    if isinstance(metrics, dict):
                        for k in metrics:
                            log_metric(metrics[k], base_string + '{}_'.format(k))
                    else:
                        neptune.log_metric(base_string, metrics)
            for k in d:
                if k in metrics_to_be_logged:
                    log_metric(d[k], base_string='{}_'.format(k))

        return d