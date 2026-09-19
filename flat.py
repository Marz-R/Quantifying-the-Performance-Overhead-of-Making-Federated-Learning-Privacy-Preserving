import torch

class Flat:
    def __init__(self, state_dict):
        self.shared_keys = []
        self.local_keys = []
        self.shapes = []
        self.numels = []
        self.offsets = []
        self.dtypes = []

        offset = 0
        for key, value in state_dict.items():
            # skipping the int counter in batchNorm so it doesn't break secret_share._to_ring()
            if not value.is_floating_point():
                self.local_keys.append(key)
                continue

            self.shared_keys.append(key)
            self.shapes.append(value.shape)
            self.numels.append(value.numel()) # num_elements
            self.dtypes.append(value.dtype)
            self.offsets.append(offset)
            offset += value.numel()

        self.uniform_dtype = self.dtypes[0] if self.dtypes and len(set(self.dtypes)) == 1 else None
        self.total = offset


    def matches(self, state_dict):
        matched_len = len(self.shared_keys) + len(self.local_keys) == len(state_dict)
        matched_keys = all(key in state_dict for key in self.shared_keys)

        return matched_len and matched_keys


    def flatten(self, state_dict, device, dtype=torch.float64):
        # float64 -> adss, bc _to_ring() moves it to float32 anyway
        # float32 -> plaintext
        if self.uniform_dtype is not None:
            flat = torch.cat([state_dict[key].reshape(-1) for key in self.shared_keys])
        else:
            flat = torch.cat([state_dict[key].reshape(-1).to(dtype) for key in self.shared_keys])

        return flat.to(dtype=dtype, device=device)


    def unflatten(self, flat, local_state_dict):
        state_dict = {key: local_state_dict[key] for key in self.local_keys}

        # optimize dtype convertion if uniform dtype
        if self.uniform_dtype is not None:
            flat = flat.to(self.uniform_dtype)

        for key, shape, dtype, offset, numel in zip(self.shared_keys, self.shapes, self.dtypes, self.offsets, self.numels):
            chunk = flat[offset : offset + numel].reshape(shape)
            state_dict[key] = chunk if chunk.dtype == dtype else chunk.to(dtype) # dtype conversion if not uniform dtype

        return state_dict
