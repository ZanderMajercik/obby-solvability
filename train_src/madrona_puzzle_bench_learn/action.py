import torch
from torch.distributions.categorical import Categorical

class DiscreteActionDistributions:
    def __init__(self, actions_num_buckets, logits = None):
        self.actions_num_buckets = actions_num_buckets

        # Guard against non-finite logits: Categorical / multinomial raise
        # "probability tensor contains inf/nan". A rare degenerate level can make
        # the actor output inf. nan -> 0 and clamping to a finite range keep
        # sampling valid without stopping a long run.
        if logits is not None:
            logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0).clamp_(-50.0, 50.0)

        self.dists = []
        self.unnormalized_logits = []
        cur_bucket_offset = 0

        for num_buckets in self.actions_num_buckets:
            sliced_logits = \
                logits[:, cur_bucket_offset:cur_bucket_offset + num_buckets]

            self.dists.append(Categorical(
                logits=sliced_logits, validate_args=False))
            self.unnormalized_logits.append(sliced_logits)
            cur_bucket_offset += num_buckets

    def best(self, out):
        actions = [dist.probs.argmax(dim=-1) for dist in self.dists]
        torch.stack(actions, dim=1, out=out)

    def sample(self, actions_out, log_probs_out):
        actions = [dist.sample() for dist in self.dists]
        log_probs = [dist.log_prob(action) for dist, action in zip(self.dists, actions)]

        torch.stack(actions, dim=1, out=actions_out)
        torch.stack(log_probs, dim=1, out=log_probs_out)

    def sample_out(self):
        actions = [dist.sample() for dist in self.dists]
        log_probs = [dist.log_prob(action) for dist, action in zip(self.dists, actions)]

        return torch.stack(actions, dim=1), torch.stack(log_probs, dim=1)

    def action_stats(self, actions):
        log_probs = []
        entropies = []
        for i, dist in enumerate(self.dists):
            log_probs.append(dist.log_prob(actions[:, i]))
            entropies.append(dist.entropy())

        return torch.stack(log_probs, dim=1), torch.stack(entropies, dim=1)

    def probs(self):
        return [dist.probs for dist in self.dists]
