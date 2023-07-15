import gc
import mmap
import os
from tempfile import mkstemp

import torch


class TensorCache:
    """
    Temporary file containing a list of read-only tensors.

    The list can't be modified after reading from it.
    """

    def __init__(self, dir=None):
        self.fd, name = mkstemp(dir=dir)
        # Automatically delete when the file descriptor is closed
        os.unlink(os.path.join(dir, name))

        self.map = None
        self.tensors = []  # (start, end, dtype, size)
        self.START, self.END, self.DTYPE, self.SIZE = 0, 1, 2, 3

    def append(self, tensor):
        assert self.map is None, "Can't append tensors after reading from file"

        start = self.tensors[-1][self.END] if len(self.tensors) > 0 else 0
        end = start + tensor.element_size() * tensor.numel()
        dtype, size = tensor.dtype, tensor.size()

        b = bytes(tensor.detach().cpu().contiguous().numpy())
        assert len(b) == end - start, (len(b), tensor.element_size(), tensor.numel())

        os.lseek(self.fd, start, os.SEEK_SET)
        os.write(self.fd, b)
        self.tensors.append((start, end, dtype, size))

    def cat_to_last(self, tensor):
        assert self.map is None, "Can't concatenate to last tensor after reading from file"
        assert len(self.tensors) > 0, "No tensor to concatenate to"

        last = self.tensors[-1]
        assert last[self.DTYPE] == tensor.dtype, (
            "Must have same type",
            last[self.DTYPE],
            tensor.dtype,
        )
        assert len(last[self.SIZE]) == len(tensor.size()), (
            "Must have same number of dimensions",
            last[self.SIZE],
            tensor.size(),
        )
        for dim in range(1, len(last[self.SIZE])):
            assert last[self.SIZE][dim] == tensor.size(dim), (
                "Must have same shape in dimensions after first",
                last[self.SIZE],
                tensor.size(),
            )
        cat_size = (last[self.SIZE][0] + tensor.size(0), *last[self.SIZE][1:])

        self.append(tensor)
        cat_end = self.tensors[-1][self.END]  # Appended tensor end
        self.tensors.pop()  # Un-append tensor
        self.tensors[-1] = (
            self.tensors[-1][self.START],
            cat_end,
            self.tensors[-1][self.DTYPE],
            cat_size,
        )

    def _map(self):
        if self.map is None:
            os.lseek(self.fd, 0, os.SEEK_SET)
            os.fsync(self.fd)
            self.map = mmap.mmap(self.fd, self.tensors[-1][self.END], access=mmap.ACCESS_READ)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, i_tensor):
        """Returns read-only tensor with appended data."""

        self._map()
        info = self.tensors[i_tensor]
        tensor = torch.frombuffer(
            self.map[info[self.START] : info[self.END]], dtype=info[self.DTYPE]
        ).view(info[self.SIZE])
        return tensor

    def __iter__(self):
        # OPTIM
        for i in range(len(self.tensors)):
            yield self[i]

    def close(self):
        if self.map is not None:
            self.map.close()
            self.map = None

        os.close(self.fd)
        self.fd = None


if __name__ == "__main__":
    print(torch.cuda.memory_allocated() / 1e6, "x")
    x = torch.ones(int(1e7), dtype=torch.float32, device="cuda")

    print(torch.cuda.memory_allocated() / 1e6, "c")
    c = TensorCache(dir="./tmp")

    print(torch.cuda.memory_allocated() / 1e6, "append")
    c.append(x)

    print(torch.cuda.memory_allocated() / 1e6, "x = None")
    x = None

    print(torch.cuda.memory_allocated() / 1e6, "collect")

    print(torch.cuda.memory_allocated() / 1e6, "empty")
    torch.cuda.empty_cache()

    print(torch.cuda.memory_allocated() / 1e6, "x")
    x = c[0].cuda()

    print(torch.cuda.memory_allocated() / 1e6, "close")
    c.close()

    print(torch.cuda.memory_allocated() / 1e6, "c = None")
    c = None

    print(torch.cuda.memory_allocated() / 1e6, "collect")

    print(torch.cuda.memory_allocated() / 1e6, "exit")
