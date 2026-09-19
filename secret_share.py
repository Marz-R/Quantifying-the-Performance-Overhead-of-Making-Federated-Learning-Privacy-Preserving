# n-out-of-n additive secret sharing over the integer ring Z_(2^k)
import secrets
import torch
from flat import Flat

# Supported ring widths
RING_DTYPES = {32: torch.int32, 64: torch.int64}

DEFAULT_RING_BITS = 32 # k
DEFAULT_MULTIPLIER = 1 << 22  # shift 22 bits to left; float -> int


class SecretShare:
    def __init__(self, multiplier=DEFAULT_MULTIPLIER, ring_bits=DEFAULT_RING_BITS,
        device=None, seed=None):
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if ring_bits not in RING_DTYPES:
                raise ValueError(f"ring_bits must be one of {sorted(RING_DTYPES)}, got {ring_bits}")
        self.ring_bits = ring_bits
        self.ring_dtype = RING_DTYPES[ring_bits]

        self._bytes_per_element = ring_bits // 8
        self._ring_max = 1 << (ring_bits - 1)

        self.multiplier = float(multiplier)

        self._generator = torch.Generator(device=self.device)
        self._fixed_seed = seed is not None #bool to control reseeding -> if false, reseed every sync for mask gen
        self.reseed(seed)

        self._layout = None


    def reseed(self, seed=None):
        self._generator.manual_seed(secrets.randbits(63) if seed is None else seed)


    def share(self, secret, num_shares):
        # splits a state_dict into n flat ring vectors
        if num_shares < 2:
            raise ValueError(f"num_shares must be at least 2 to hide the secret, got {num_shares}")

        if self._layout is None or not self._layout.matches(secret):
            self._layout = Flat(secret) # skippable if same model is cached in prev sync

        if not self._fixed_seed:
            self.reseed()

        flattened = self._layout.flatten(secret, self.device)
        ring = self._to_ring(flattened, num_shares) # -> integer

        masks = self._random_ring(num_shares - 1, self._layout.total) # gen all masks parallelly in a block
        last = ring - masks.sum(dim=0, dtype=self.ring_dtype)

        shares = []
        for i in range(num_shares - 1):
            shares.append(masks[i].clone()) 
        shares.append(last)

        return shares


    def reconstruct(self, shares):
        # adds shares together to a partial sum
        stacked = torch.stack([share.to(device=self.device, dtype=self.ring_dtype) for share in shares])

        return stacked.sum(dim=0, dtype=self.ring_dtype)


    def average(self, ring_totals, local_state_dict):
        # aggregates all partial sums and unflattens it back into state_dict
        reconstructed = self.reconstruct(ring_totals)
        flat = self._from_ring(reconstructed) / len(ring_totals)
        avg_state_dict = self._layout.unflatten(flat, local_state_dict)

        return avg_state_dict


    def _random_ring(self, num_masks, numel):
        size = (num_masks, numel, self._bytes_per_element)
        raw_bytes = torch.randint(0, 256, size, dtype=torch.uint8, device=self.device, generator=self._generator)

        return raw_bytes.view(self.ring_dtype).squeeze(-1)


    def _to_ring(self, flat, num_shares):
        scaled = torch.round(flat * self.multiplier)

        # predict & check if reconstructed partial sums overflows otherwise it proceeds without raising error
        largest = scaled.abs().max().item() * max(num_shares, 1)
        if largest >= self._ring_max:
            raise OverflowError(
                f"reconstruction total would reach {largest:.3e}, ring holds {self._ring_max:.3e}; "
                f"lower multiplier (currently {self.multiplier:.0f}) or use ring_bits=64"
            )
        
        return scaled.to(self.ring_dtype)


    def _from_ring(self, flat):
        return flat.to(torch.float64) / self.multiplier
 
    @property
    def layout(self):
        return self._layout

    

    
