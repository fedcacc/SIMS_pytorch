from utils.utils import MODE
import numpy as np


class StoreJointOptimized:
    """Callable to store episodes in SIMS reservoir buffer"""
    def __init__(self, buffers):
        self.buffers = buffers
        self.team_policies = ["policy_t1", "policy_t2"]  # TODO (fede): pass team policies from configs

    def __call__(self, batch):
        policy_batches = [batch.policy_batches[p] for p in self.team_policies]
        if len(policy_batches) == 0:
            return
        episodes = np.unique(policy_batches[0]["eps_id"])
        for ep in episodes:
            indexes = [np.where(p["eps_id"] == ep)[0] for p in policy_batches]
            lengths = [len(p["mode"][indexes[i]]) for i, p in enumerate(policy_batches)]
            rollout_len = min(lengths)
            for sample in range(rollout_len):
                if policy_batches[0]["mode"][indexes[0][sample]] == MODE.best_response.value:
                    obs = [p["obs"][indexes[i][sample]] for i, p in enumerate(policy_batches)]
                    ac = np.array([p["actions"][indexes[i][sample]] for i, p in enumerate(policy_batches)])
                    obs = np.concatenate([obs[0]]*len(obs), axis=0)
                    self.buffers.buffers["policy_team"].add(
                        obs,
                        ac
                    )
            if len(set(lengths)) > 1:
                longer_batch = int(np.argmax(lengths))
                if policy_batches[longer_batch]["mode"][indexes[longer_batch][rollout_len]] == MODE.best_response.value:
                    obs = policy_batches[longer_batch]["obs"][indexes[longer_batch][rollout_len]]
                    obs = np.concatenate([obs]*len(policy_batches), axis=0)
                    ac = np.zeros(shape=(len(policy_batches), 1))
                    ac[longer_batch] = policy_batches[longer_batch]["actions"][indexes[longer_batch][rollout_len]]
                    self.buffers.buffers["policy_team"].add(
                        obs,
                        ac
                    )
            self.buffers.steps["policy_team"] += rollout_len


class StoreJointOptimizedSinglePolicy:
    """Callable to store episodes in SIMS reservoir buffer"""
    def __init__(self, buffers):
        self.buffers = buffers
        self.team_policy = "policy_t1"  # TODO (fede): pass team policies from configs

    def __call__(self, batch):
        # policy_batches = [batch.policy_batches[p] for p in self.team_policies]
        team_batch = batch.policy_batches[self.team_policy]
        team_batch_rows = np.array(list(team_batch.rows()))
        t1_mask = team_batch["agent_id"] == "t1"
        t1_rows = team_batch_rows[t1_mask]
        t2_rows = team_batch_rows[np.logical_not(t1_mask)]

        for t1_r, t2_r in zip(t1_rows, t2_rows):
            assert t1_r["eps_id"] == t2_r["eps_id"], "Error when unrolling batch!"
            if (t1_r["mode"] == MODE.best_response.value) and (t2_r["mode"] == MODE.best_response.value):
                obs = np.concatenate([t1_r["obs"]]*2, axis=0)
                ac = np.array([t1_r["actions"], t2_r["actions"]])
                self.buffers.buffers["policy_team"].add(
                    obs,
                    ac
                )
                self.buffers.steps["policy_team"] += 1
