from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()


def eval_probs(policy, obs_batch, mask=None):
    """Function that evaluates probs at training time TODO (fede): remove"""
    dd = {}
    # obs_shape = obs_batch.size(1)
    obs_to_eval, indexes = torch.unique(obs_batch, dim=0, return_inverse=True)
    with torch.no_grad():
        logits, _ = policy.model({'obs': obs_to_eval})
        probs = nn.functional.softmax(logits, dim=1)

    for i, o in enumerate(obs_to_eval):
        dd.update({
          "obs_{}_ac_{}_prob".format(o.detach().cpu().numpy(), ac): float(probs[i, ac])
          for ac in range(probs.size(-1))
        })

    policy.stats_dict = dd