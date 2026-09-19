from privacy_protocol import PrivacyProtocol
from secret_share import SecretShare


class AdditiveSecretSharing(PrivacyProtocol):
    def __init__(self, peers_list, multiplier=None, ring_bits=None, seed=None):
        kwargs = {}
        if multiplier is not None:
            kwargs["multiplier"] = multiplier
        if ring_bits is not None:
            kwargs["ring_bits"] = ring_bits
        if seed is not None:
            kwargs["seed"] = seed

        self.secret_share = SecretShare(**kwargs)
        self._num_peers = len(peers_list)
        self._peers_list = peers_list


    def before_send(self, weights: dict) -> dict:
        shares = self.secret_share.share(weights, self._num_peers)
        shares_dict = {}
        for peer_id, share in zip(self._peers_list, shares):
            shares_dict[peer_id] = share.cpu() # to cpu for sending

        return shares_dict


    def after_receive(self, weights: dict):
        # compute ring sum to give partial sum after phase 1
        return self.secret_share.reconstruct(list(weights.values()))


    def aggregate(self, local_partial, peers_partials, local_state_dict) -> dict:
        # compute partial sums to give final model after phase 2
        ring_totals = [local_partial] + list(peers_partials)

        return self.secret_share.average(ring_totals, local_state_dict)


    def update_peers_list(self, peers_list):
        self._peers_list = peers_list
        self._num_peers = len(peers_list)
