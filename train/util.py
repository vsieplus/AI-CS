import gc
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from hyper import HOP_LENGTH, CHART_FRAME_RATE

# avoid subdirs clutter when adding hparams with summary writer
class SummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()
        
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)

# memory reporting
def report_memory(device, show_tensors=False):
    """Simple CPU/GPU memory report."""

    if device.type == 'cuda':
        mega_bytes = 1024.0 * 1024.0
        string = device.type + 'memory (MB)'
        string += ' | allocated: {}'.format(torch.cuda.memory_allocated(device) / mega_bytes)
        string += ' | max allocated: {}'.format(torch.cuda.max_memory_allocated(device) / mega_bytes)
        string += ' | cached: {}'.format(torch.cuda.memory_cached(device) / mega_bytes)
        string += ' | max cached: {}\n'.format(torch.cuda.max_memory_cached(device)/ mega_bytes)
        print(string)

    if show_tensors:
        print('Tensor Report:\n')
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and device.type == obj.device.type: 
                print(type(obj), obj.size(), f'requires_grad: {obj.requires_grad}')
                if(obj.size() == torch.Size([2])):
                    print(obj.data)
        print('\n')

def convert_chartframe_to_melframe(frame, sample_rate, hop_length=HOP_LENGTH, chart_frame_rate=CHART_FRAME_RATE):
    """convert chart frame #s (10ms) -> audio frame #s [from mel spectrogram representation]
	   melframe = round((sample_rate * secs) / hop_length)
	   -> secs = (hop * melframe) / sample_rate

       chart_frame_rate = # of chart frames per second (e.g. 100 -> 10 ms chart frames)
    """
    frame_secs = frame / chart_frame_rate
    return frame_secs * sample_rate / hop_length