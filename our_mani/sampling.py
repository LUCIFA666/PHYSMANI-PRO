"""Resume shuffled HDF5 training by optimizer update, without decoding skipped RGB."""
import torch


class ResumeableRandomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_size, batch_size, *, start_update=0, seed=0):
        self.size, self.batch_size, self.seed = dataset_size, batch_size, seed
        self.usable = dataset_size//batch_size*batch_size
        if not self.usable or start_update < 0:
            raise ValueError('invalid resumable sampling configuration')
        consumed = start_update*batch_size
        self.epoch, self.offset = divmod(consumed, self.usable)

    def __len__(self):
        return self.usable-self.offset

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()[:self.usable]
        offset = self.offset
        self.epoch, self.offset = self.epoch+1, 0
        yield from order[offset:]
