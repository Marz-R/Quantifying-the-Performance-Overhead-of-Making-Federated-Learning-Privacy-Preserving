from privacy_protocol import PrivacyProtocol
from secret_share import SecretShare
import torch

class AdditiveSecretSharing(PrivacyProtocol):
    def __init__(self, peers_list):
        self.secret_share = SecretShare()
        self._shares = {}
        self._num_peers = len(peers_list)
        self._peers_list = peers_list


    def before_send(self, weights: dict) -> dict:
        # generate shares
        total_shares = {}
        for k, v in weights.items():
            shares = self.secret_share.share(v, self._num_peers)
            total_shares[k] = shares

        for i in range(self._num_peers):
            peer_id = self._peers_list[i]
            peer_shares = {k: total_shares[k][i] for k in total_shares.keys()}
            self._shares[peer_id] = peer_shares

        return self._shares


    def after_receive(self, weights: dict) -> dict:
        # reconstruct weights from shares
        reconstructed_weights = {}
        partial_weights_lists = {}
        for peer_id in weights.keys():
            partial_weight = weights[peer_id]
            for layer, value in partial_weight.items():
                if layer not in partial_weights_lists:
                    partial_weights_lists[layer] = []
                partial_weights_lists[layer].append(value)

        for layer, shares in partial_weights_lists.items():
            reconstructed_weights[layer] = self.secret_share.reconstruct(shares).to(dtype=torch.float32)

        return reconstructed_weights


    def update_peers_list(self, peers_list):
        self._peers_list = peers_list
        self._num_peers = len(peers_list)
