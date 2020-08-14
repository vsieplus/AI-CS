import gc
import torch

# memory reporting
def report_memory(name='', show_tensors=False):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + 'memory (MB)'
    string += ' | allocated: {}'.format(torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | cached: {}'.format(torch.cuda.memory_cached() / mega_bytes)
    string += ' | max cached: {}\n'.format(torch.cuda.max_memory_cached()/ mega_bytes)
    print(string)

    if show_tensors:
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
            except:
                pass
        print('\n')
