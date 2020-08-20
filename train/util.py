import gc
import torch

from hyper import HOP_LENGTH, CHART_FRAME_RATE

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

def convert_chartframe_to_melframe(frame, sample_rate, hop_length=HOP_LENGTH, chart_frame_rate=CHART_FRAME_RATE):
    """convert chart frame #s (10ms) -> audio frame #s [from mel spectrogram representation]
	   melframe = round((sample_rate * secs) / hop_length)
	   -> secs = (hop * melframe) / sample_rate

       chart_frame_rate = # of chart frames per second (e.g. 100 -> 10 ms chart frames)
    """
    frame_secs = frame / chart_frame_rate
    return frame_secs * sample_rate / hop_length