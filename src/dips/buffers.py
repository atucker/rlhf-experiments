import torch
import wandb


def to_device(inputs, device):
    return dict((key, value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in inputs.items())


def fuse_dicts(separator='/', **kwargs):
    """Go from A={X: ...}, B={X: ...} to {A/X: ..., B/X: ...}"""
    ans = {}
    for key, value in kwargs.items():
        if not isinstance(value, dict):
            ans[key] = value
        else:
            for key2, value2 in fuse_dicts(**value).items():
                assert not isinstance(value2, dict)
                ans[f"{key}{separator}{key2}"] = value2
    return ans


def split_dicts(batch, keys, unpack=False, censor=None, remainder=False):
    """
    Go from {A/X: ..., B/X: ...} to {A: {X: ...}, B: {X: ...}}

    if unpack is True, then unpack the dict into the given order
    if censor is True, then don't encode those keys (X, not A)
    """
    ans = {}
    found = set()
    for search_key in keys:
        key_dict = {}
        search_key = f"{search_key}/"
        for data_key, value in batch.items():
            if search_key == data_key[:len(search_key)]:
                key = data_key[len(search_key):]
                if censor is None or key not in censor:
                    key_dict[key] = value
                found |= {data_key}
        ans[search_key[:-1]] = key_dict

    if remainder:
        remainder = {}
        for key, value in batch.items():
            if key not in found:
                remainder[key] = value
        if unpack:
            return (ans[key] for key in keys), remainder
        else:
            return ans, remainder
    else:
        if unpack:
            return (ans[key] for key in keys)
        else:
            return ans


def batch_to_list(batch):
    n = None
    for key, value in batch.items():
        if n is None:
            n = len(value)
        else:
            assert len(value) == n, f"{key} has length {len(value)} instead of {n}\n{batch}"

    return [batch_row(batch, idx) for idx in range(n)]


def list_to_batch(inpt, return_list=False):
    ans = {}
    for key in inpt[0].keys():
        values = [row[key] for row in inpt]
        if return_list:
            ans[key] = values
        else:
            ans[key] = torch.stack(values) if not isinstance(values[0], str) else values
    return ans


def gather_dict(accelerator, data):
    # Put everything on the GPU
    for key in data.keys():
        if not isinstance(data[key], torch.Tensor):
            data[key] = torch.tensor(data[key]).float().to(accelerator.device)
    return accelerator.gather(data)


def gather_uneven_dict(accelerator, data):
    # Figure out lengths, and pad the data
    local_n = torch.tensor(dict_length(data)).to(accelerator.device)
    lengths = accelerator.gather(local_n).cpu().numpy().tolist()
    if all([_ == lengths[0] for _ in lengths]):
        return accelerator.gather(data)

    if accelerator.is_main_process:
        print(f"Gathering lengths: {lengths}")

    max_n = max(lengths)
    for key, value in data.items():
        pad = torch.zeros(max_n - local_n, *value.shape[1:]).to(accelerator.device).type(value.dtype)
        data[key] = torch.cat([data[key], pad])
    gathered = accelerator.gather(data)

    # Slice the data back into the original lengths to remove the padding
    dicts = []
    start = 0
    for n in lengths:
        end = start + n
        process_data = slice_dict(gathered, start, end)
        assert dict_length(process_data) == n
        dicts.append(process_data)
        start = start + max_n

    # Accumulate the dictionaries into our answer
    ans = {}
    for data in dicts:
        ans = accumulate_dict(ans, data)
    return ans


def reduce_dict(data, fn=torch.mean):
    ans = {}
    for key in data.keys():
        ans[key] = fn(data[key].float()).detach().cpu().item()
    return ans


def gather_print(accelerator, data):
    data = reduce_dict(gather_dict(accelerator, data))
    if accelerator.is_main_process:
        wandb.log(data)
        print(data)


def accumulate_dict(base, data):
    if len(data.keys()) == 0:
        return base

    ans = {}
    for key, value in data.items():
        if key not in base:
            ans[key] = value
        elif isinstance(value, list):
            ans[key] = base[key] + value
        elif isinstance(value, torch.Tensor):
            ans[key] = torch.cat([base[key], value], axis=0)
        elif isinstance(value, dict):
            ans[key] = accumulate_dict(base[key], value)
        else:
            assert False, f"I don't know how to handle {value}"

    return ans


def reduce_dict(data, fn=torch.mean):
    ans = {}
    for key in data.keys():
        ans[key] = fn(data[key].float()).detach().cpu().item()
    return ans


def dict_length(data):
    if isinstance(data, list) or isinstance(data, tuple):
        return len(data)
    elif isinstance(data, torch.Tensor):
        return data.shape[0]
    elif isinstance(data, dict):
        ans = None
        for key, value in data.items():
            if ans is None:
                ans = dict_length(value)
            else:
                assert ans == dict_length(value), f"{key}: {ans} != {dict_length(value)}"
        return ans or 0


def slice_dict(data, start, end):
    if isinstance(data, list) or isinstance(data, tuple):
        return data[start:end]
    elif isinstance(data, torch.Tensor):
        return data[start:end]
    elif isinstance(data, dict):
        ans = {}
        for key, value in data.items():
            ans[key] = slice_dict(value, start, end)
        return ans


def cat_dict(d1, d2, axis=1):
    ans = {}
    for key, value in d2.items():
        ans[key] = torch.cat([d1[key], value], axis=axis)
    return ans


def distributed_iter_dict(accelerator, batch, batch_size):
    start = batch_size * accelerator.process_index
    while start + batch_size <= dict_length(batch):
        end = start + batch_size
        yield slice_dict(batch, start, end)
        start += batch_size * accelerator.num_processes


def remove_strs(data):
    ans = {}
    for key, value in data.items():
        if isinstance(value, dict):
            ans[key] = remove_strs(value)
        elif isinstance(value[0], str):
            pass
        else:
            ans[key] = value
    return ans


class DistributedBuffer:
    def __init__(self, accelerator) -> None:
        self._buffer = {}
        self._accelerator = accelerator
        self._local_n = 0
        self._distributed_n = 0

    def add(self, data):
        data = remove_strs(data)
        self._buffer = accumulate_dict(self._buffer, data)
        self._local_n += dict_length(data)
        assert self._local_n == dict_length(self._buffer)
        count = torch.tensor(self._local_n).to(self._accelerator.device)
        self._distributed_n = torch.sum(self._accelerator.gather(count)).cpu().item()
        if self._accelerator.is_main_process:
            print(f"Buffer size: {self._distributed_n}")

    def __len__(self):
        return self._distributed_n

    def all_ready(self):
        ready = 1 if len(self._buffer.keys()) > 0 else 0
        ready = torch.tensor(ready).to(self._accelerator.device)
        return torch.all(self._accelerator.gather(ready) > 0)

    def gather_n(self, batch_size):
        """Return batch_size items to every process, and put the remainder on the main process"""
        assert batch_size <= self._distributed_n, f"Wanted {batch_size}, have {self._distributed_n}"
        print(f"Getting {batch_size} from {self._distributed_n} on thread {self._accelerator.process_index}: local {self._local_n}")
        gathered = gather_uneven_dict(self._accelerator, self._buffer)
        assert dict_length(gathered) == self._distributed_n, f"{dict_length(gathered)} != {self._distributed_n}"

        ans = slice_dict(gathered, 0, batch_size)
        remainder = slice_dict(gathered, batch_size, None)

        if self._accelerator.is_main_process:
            self._buffer = remainder
            self._local_n = dict_length(remainder)
            print(f"Remainder size: {self._local_n}")
        else:
            self._buffer = slice_dict(ans, batch_size, None)
            assert dict_length(self._buffer) == 0
            self._local_n = 0
            print(f"Thread {self._accelerator.process_index} cleared buffer")

        count = torch.tensor(self._local_n).to(self._accelerator.device)
        self._distributed_n = torch.sum(self._accelerator.gather(count)).cpu().item()
        return ans