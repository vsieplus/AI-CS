import gc
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from hyper import HOP_LENGTH, CHART_FRAME_RATE, CLSTM_SAVE, SRNN_SAVE, CHECKPOINT_SAVE

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
                    print(obj)
        print('\n')

def convert_chartframe_to_melframe(frame, sample_rate, hop_length=HOP_LENGTH, chart_frame_rate=CHART_FRAME_RATE):
    """convert chart frame #s (10ms) -> audio frame #s [from mel spectrogram representation]
	   melframe = round((sample_rate * secs) / hop_length)
	   -> secs = (hop * melframe) / sample_rate

       chart_frame_rate = # of chart frames per second (e.g. 100 -> 10 ms chart frames)
    """
    frame_secs = frame / chart_frame_rate
    return frame_secs * sample_rate / hop_length

def convert_melframe_to_secs(melframe, sample_rate, hop_length=HOP_LENGTH):
    return (hop_length * melframe) / sample_rate
    
## Model saving/loading #####################################    
    
def save_checkpoint(epoch, curr_epoch_batch, best_placement_valid_loss, best_placement_precision,
                    best_selection_valid_loss, train_clstm, train_srnn, save_dir):
    out_path = os.path.join(save_dir, CHECKPOINT_SAVE)
    
    print(f'\tSaving checkpoint to {out_path}')
    torch.save({
        'epoch': epoch,
        'curr_epoch_batch': curr_epoch_batch,
        'best_placement_valid_loss': best_placement_valid_loss,
        'best_placement_precision': best_placement_precision,
        'best_selection_valid_loss': best_selection_valid_loss,
        'train_clstm': train_clstm,
        'train_srnn': train_srnn,
    }, out_path)

def save_model(model, save_dir, model_filename):
    out_path = os.path.join(save_dir, model_filename)
    print(f'\tSaving model to {out_path}')

    torch.save(model.state_dict(), out_path)

def load_save(save_dir, fine_tune, placement_clstm, selection_rnn, device):
    print(f'Loading checkpoint from {save_dir}...')

    clstm_path = os.path.join(save_dir, CLSTM_SAVE)
    srnn_path = os.path.join(save_dir, SRNN_SAVE)
    
    if os.path.isfile(clstm_path) and placement_clstm is not None:
        placement_clstm.load_state_dict(torch.load(clstm_path, map_location=device))
    else:
        print(f'No saved {CLSTM_SAVE} file found in {save_dir}, starting from base model')

    if os.path.isfile(srnn_path) and selection_rnn is not None:
        selection_rnn.load_state_dict(torch.load(srnn_path, map_location=device))
    else:
        print(f'No saved {SRNN_SAVE} file found in {save_dir}, starting from base model')

    checkpoint = torch.load(os.path.join(save_dir, CHECKPOINT_SAVE))

    # only restore epoch/best loss values when not fine_tuneing    
    # (i.e. give option to fine_tune some more on an already trained model)       
    if not fine_tune:
        start_epoch = checkpoint['epoch']
        start_epoch_batch = checkpoint['curr_epoch_batch']
        best_placement_valid_loss = checkpoint['best_placement_valid_loss']
        best_placement_precision = checkpoint['best_placement_precision']
        best_selection_valid_loss = checkpoint['best_selection_valid_loss']
        train_clstm = checkpoint['train_clstm']
        train_srnn = checkpoint['train_srnn']

        # use last directory under save_dir/runs
        logdirs = [d for d in os.listdir(os.path.join(save_dir, 'runs'))]
        sub_logdir = [d for d in logdirs if os.path.isdir(os.path.join(save_dir, 'runs', d))][-1]

        return (start_epoch, start_epoch_batch, best_placement_valid_loss, best_placement_precision,
                best_selection_valid_loss, train_clstm, train_srnn, sub_logdir)
    else:
        return None
