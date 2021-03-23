# utility functions for training

import gc
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

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
        best_selection_valid_loss = checkpoint['best_selection_valid_loss']
        train_clstm = checkpoint['train_clstm']
        train_srnn = checkpoint['train_srnn']
    
        new_model = 'best_placement_precision' in checkpoint
        best_placement_precision = checkpoint['best_placement_precision'] if new_model else 0

        # use last directory under save_dir/runs
        logdirs = [d for d in os.listdir(os.path.join(save_dir, 'runs'))]
        sub_logdir = [d for d in logdirs if os.path.isdir(os.path.join(save_dir, 'runs', d))][-1]

        return (start_epoch, start_epoch_batch, best_placement_valid_loss, best_placement_precision,
                best_selection_valid_loss, train_clstm, train_srnn, sub_logdir)
    else:
        return None

def collate_charts(batch):
	"""custom collate function for dataloader
		input: dict of chart objects/lengths
		output: dict of batch inputs, targets
	"""
	batch = list(filter(lambda x: x is not None, batch))

	batch_size = len(batch)

	audio_feats = []
	chart_feats = []
	levels = []

	first_frame = 100000000
	last_frame = -1

	placement_targets = []

	step_sequence = []
	step_targets = []

	# list of lengths of examples, for packing
	audio_lengths = []
	step_sequence_lengths = []

	for chart in batch:
		# skip empty charts
		if chart.n_steps == 0:
			continue

		# transpose channels/timestep so timestep comes first for pad_sequence()
		audio_feats.append(chart.get_audio_feats().transpose(0, 1))
		chart_feats.append(torch.tensor(chart.chart_feats).unsqueeze(0))
		placement_targets.append(chart.placement_targets)

		levels.append(chart.level)

		# already tensors
		step_sequence.append(torch.cat((chart.step_features, chart.step_time_features), dim=-1))
		step_targets.append(chart.step_targets)

		if chart.first_frame < first_frame:
			first_frame = chart.first_frame
		
		if chart.last_frame > last_frame:
			last_frame = chart.last_frame

		audio_lengths.append(audio_feats[-1].size(0))
		step_sequence_lengths.append(step_sequence[-1].size(0))

	# transpose timestep/channel dims back => [batch, channels, max audio frames, freq]
	audio_feats = pad_sequence(audio_feats, batch_first=True, padding_value=PAD_IDX).transpose(1, 2)
	chart_feats = torch.cat(chart_feats, dim=0) # [batch, chart_feats]

	# [batch, max audio frames]
	placement_targets = pad_sequence(placement_targets, batch_first=True, padding_value=PAD_IDX)

	# [batch, max arrow seq len, arrow features] / [batch, max arrow seq len]
	step_sequence = pad_sequence(step_sequence, batch_first=True, padding_value=PAD_IDX)
	step_targets = pad_sequence(step_targets, batch_first=True, padding_value=PAD_IDX)

	return {'audio_feats': audio_feats,
			'chart_feats': chart_feats,
			'chart_levels': levels,
			'audio_lengths': audio_lengths,
			'placement_targets': placement_targets,
			'first_step_frame': first_frame,
			'last_step_frame': last_frame,
			'step_sequence': step_sequence,
			'step_sequence_lengths': step_sequence_lengths,
			'step_targets': step_targets,
	}

   
def get_dataloader(dataset, batch_size, indices):
    index_sampler = SubsetRandomSampler(indices)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_charts, sampler=index_sampler)
