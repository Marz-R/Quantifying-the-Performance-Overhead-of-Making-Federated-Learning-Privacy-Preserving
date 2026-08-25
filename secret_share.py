# m-out-of-n additive secret sharing

import torch

class SecretShare:
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def share(self, secret, num_shares):
        """
        Splits a secret into n additive shares.
        Args:
            secret (Tensor): The secret to share.
            num_shares (int): Number of shares to create.
        Returns:
            List[Tensor]: A list of n shares such that their sum modulo PRIME equals the secret.
        """

        shares = []
        secret = secret.to(dtype=torch.float32, device=self.device) # to device
        sum_shares = torch.zeros_like(secret, dtype=torch.float32, device=self.device)

        for _ in range(num_shares - 1):
            share = torch.rand(secret.shape, device=self.device)
            shares.append(share) 
            sum_shares += share

        final_share = (secret - sum_shares)
        shares.append(final_share)
        
        return shares

    def reconstruct(self, shares):
        """
        Reconstructs the secret from additive shares.
        Args:
            shares (List[Tensor]): List of n shares.
        Returns:
            Tensor: The reconstructed secret.
        """

        shares = [share.to(dtype=torch.float32, device=self.device) for share in shares] # to device
        secret = torch.zeros_like(shares[0], dtype=torch.float32, device=self.device)

        for share in shares:
            secret += share

        return secret