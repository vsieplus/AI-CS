# utility classes for loading step charts

import json
import os
import re

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, Sampler, random_split
from torch.nn.utils.rnn import pad_sequence

from pathlib import Path
from joblib import Memory

from extract_audio_feats import extract_audio_feats, load_audio
from hyper import PAD_IDX, SEED, N_CHART_TYPES, N_LEVELS, CHART_FRAME_RATE, NUM_ARROW_STATES
from util import convert_chartframe_to_melframe

# cache dataset tensors/other values https://discuss.pytorch.org/t/cache-datasets-pre-processing/1062/8
ABS_PATH = str(Path(__file__).parent.absolute())
CACHE_DIR = os.path.join(ABS_PATH, '.dataset_cache/')

memory = Memory(CACHE_DIR, verbose=0, compress=True)
# to reset cache: memory.clear(warn=False)

# add caching to extract_audio_feats functions
load_audio = memory.cache(load_audio)
extract_audio_feats = memory.cache(extract_audio_feats)

# returns splits of the given (torch) dataset; assumes 3 way split
def get_splits(dataset):
	split_sizes = []
	for i, split in enumerate(dataset.splits):
		if i == len(dataset.splits) - 1:
			split_sizes.append(len(dataset) - sum(split_sizes))
		else:
			split_sizes.append(round(split * len(dataset)))
	train, valid, test = random_split(dataset, split_sizes, generator=torch.Generator().manual_seed(SEED))
	return train, valid, test

def collate_charts(batch):
	"""custom collate function for dataloader
		input: dict of chart objects/lengths
		output: dict of batch inputs, targets
	"""
	batch_size = len(batch)

	audio_feats = []
	chart_feats = []
	levels = []

	step_placements = []

	# [batch, chart frames (variable)] (just a list, not a tensor)
	# records which audio frames recorded in chart data
	step_frames = []

	step_sequence = []
	step_targets = []

	# list of lengths of examples, for packing
	audio_lengths = []
	step_sequence_lengths = []

	for chart in batch:
		# transpose channels/timestep so timestep comes first
		audio_feats.append(chart.get_audio_feats().transpose(0, 1))
		chart_feats.append(torch.tensor(chart.chart_feats).unsqueeze(0))

		levels.append((chart_feats[-1][0, 2:] == 1).nonzero(as_tuple=False).flatten().item() + 1)

		step_placements.append(torch.tensor(chart.step_placements))
		step_frames.append(chart.step_frames)

		# already tensors
		step_sequence.append(chart.step_sequence)
		step_targets.append(chart.step_targets)

		audio_lengths.append(audio_feats[-1].size(0))
		step_sequence_lengths.append(step_sequence[-1].size(0))

	# transpose timestep/channel dims back => [batch, channels, max audio frames, freq]
	audio_feats = pad_sequence(audio_feats, batch_first=True, padding_value=PAD_IDX).transpose(1, 2)
	chart_feats = torch.cat(chart_feats, dim=0) # [batch, chart_feats]

	# [batch, max arrow seq len (<= max_timesteps), arrow features]
	step_sequence = pad_sequence(step_sequence, batch_first=True, padding_value=PAD_IDX)
	step_targets = pad_sequence(step_targets, batch_first=True, padding_value=PAD_IDX)
	
	# [batch, max chart frames (10ms)] if chart frame had a hit or not
	step_placements = pad_sequence(step_placements, batch_first=True, padding_value=PAD_IDX)

    # get chart frame numbers of step placements, [batch, placements (variable)]
	step_placement_frames, first_frame, last_frame = get_step_placement_frames(step_placements, step_frames, batch_size)
	
	# for each batch example, consider all chart frames with a step,
	# and set target to '1' at corresponding melframe
	placement_targets = []
	for b in range(batch_size):
		targets = torch.zeros(audio_lengths[b], dtype=torch.long)

		sample_rate = batch[b].song.sample_rate
		mel_placement_frames = [convert_chartframe_to_melframe(f, sample_rate) 
								for f in step_placement_frames[b]]

		targets[mel_placement_frames] = 1
		placement_targets.append(targets)

	# [batch, max audio frames]
	placement_targets = pad_sequence(placement_targets, batch_first=True, padding_value=PAD_IDX)

	return {'audio_feats': audio_feats,
			'audio_lengths': audio_lengths,
			'first_step_frame': first_frame,
			'last_step_frame': last_frame,
			'chart_feats': chart_feats,
			'chart_levels': levels,
			'placement_targets': placement_targets,
			'step_sequence': step_sequence,
			'step_sequence_lengths': step_sequence_lengths,
			'step_targets': step_targets,
		}

class StepchartDataset(Dataset):
	"""Dataset of step charts"""

	# dataset_json: path to json file with dataset metadata
	def __init__(self, dataset_json):
		assert(os.path.isfile(dataset_json))
		with open(dataset_json, 'r') as f:
			metadata = json.loads(f.read())

		self.name = metadata['dataset_name']
		self.songtypes = metadata['song_types']

		self.chart_type = metadata['chart_type']
		self.step_artists = metadata['step_artists']
		self.chart_difficulties = metadata['chart_difficulties']
		self.min_level = metadata['min_chart_difficulty']
		self.max_level = metadata['max_chart_difficulty']

		self.permutations = metadata['permutations'] + [None]
		
		self.splits = metadata['splits']

		# the actual training samples
		self.load_charts(metadata['json_fps'])
		self.print_summary()

	def __len__(self):
		return len(self.charts)

	def __getitem__(self, idx):
		return self.charts[idx]

	def print_summary(self):
		print(f'Dataset name: {self.name}')
		print(f'Chart Type: {self.chart_type}')
		print(f'Total # charts: {self.__len__()}')
		if self.chart_difficulties:
			print(f'Chart levels: {self.chart_difficulties}')
		else:
			print(f'Minimum chart level: {self.min_level}')
			print(f'Maximum chart level: {self.max_level}')
		print(f'Chart Permutations: {self.permutations}')

	# filter/load charts
	def load_charts(self, json_fps):
		print('Loading charts...')
		self.songs = {}
		self.charts = []
		for fp in tqdm(json_fps):
			with open(fp, 'r') as f:
				attrs = json.loads(f.read())
			
			# check attrs for each chart to see if we should add it to the dataset
			chart_indices = self.charts_to_include(attrs)

			if not chart_indices:
				continue

			# .ssc (may contain multiple charts/same song) or .ucs (always 1 chart)    
			orig_filetype = attrs['chart_fp'].split('.')[-1]

			# create new song if needed
			song_name = attrs['title']
			if song_name not in self.songs:
				self.songs[song_name] = Song(attrs['music_fp'], song_name,
					attrs['artist'], attrs['genre'], attrs['songtype'])

			for chart_idx in chart_indices:
				for permutation in self.permutations:
					self.charts.append(Chart(attrs['charts'][chart_idx],
						self.songs[song_name], orig_filetype, permutation))
		
		print('Done loading!')
	
	# determine which charts in the given attrs belongs in this dataset
	# return list of indices, w/length between 0 <= ... <= len(attrs['charts'])
	def charts_to_include(self, attrs):
		chart_indices = []

		valid_songtype = self.songtypes and attrs['songtype'] in self.songtypes

		if not valid_songtype:
			return

		for i, chart_attrs in enumerate(attrs['charts']):
			valid_type = chart_attrs['stepstype'] == self.chart_type

			if not valid_type:
				continue

			chart_level = int(chart_attrs['meter'])
			if self.chart_difficulties:
				valid_level = chart_level in self.chart_difficulties
			else:
				valid_level = (self.min_level <= chart_level and chart_level <= self.max_level)

			if not valid_level:
				continue

			valid_author = (not self.step_artists or self.step_artists
				and chart_attrs['credit'] in self.step_artists)

			if valid_author:
				chart_indices.append(i)

		return chart_indices

class Song:
	"""A song object, corresponding to some audio file. May be relied on by
	multiple charts."""

	def __init__(self, audio_fp, title, artist, genre, songtype):
		self.audio_fp = audio_fp
		self.title = title
		self.artist = artist
		self.genre = genre
		self.songtype = songtype

		waveform, self.sample_rate = load_audio(audio_fp)

		# shape [3, ?, 80]
		self.audio_feats = extract_audio_feats(waveform, self.sample_rate)

CHART_PERMUTATIONS = {
	'pump-single': {
	    #normal:        '01234'
		'flip':         '43210',
		'mirror':       '34201',
		'flip_mirror':  '10243'
	},

	'pump-double': {
		#normal:        '0123456789'
		'flip':         '9876543210',
		'mirror':       '9875643201',
		'flip_mirror':  '1023465789'
	}
}

STEP_PATTERNS = {
	'ucs': re.compile('[XMHW]'),
	'ssc': re.compile('[1-3]')
}

UCS_SSC_DICT = {
	'.': '0',   # no step
	'X': '1',   # normal step
	'M': '2',   # start hold
	'H': '0',   # hold (0 between '2' ... '3' in ssc)
	'W': '3',   # release hold
}

# symbols used in our representation
SSC_OFF_SYMBOL = 0		# step is not activated
SSC_STEP_SYMBOL = 1		# there is a lone step, or a start of a hold
SSC_HOLD_SYMBOL = 2		# the step is currently being held down
SSC_RELEASE_SYMBOL = 3	# the step is released (end of a hold)
SSC_NUM_SYMBOLS = 4

# convert ucs steps to ssc (text)
@memory.cache
def ucs_to_ssc(steps):
	ssc_steps = ''
	for note in steps:
		ssc_steps += UCS_SSC_DICT[note]
	return ssc_steps

# convert a sequence of steps ['00100', '10120', ...] -> input tensor
@memory.cache
def sequence_to_tensor(sequence):
	# shape [abs # of frames, 4 x # arrows (20 for single, 40 for double)]
	#   (for each arrow, mark 1 of 4 possible states - off, step, hold, release)
	# eg. ['10002', '01003'] -> [[0, 1, 0, 0, 0, 0, 0, 0, ..., 0, 0, 1, 0] 
	#                             -downleft-   -upleft- ....   -downright-
	#                            [0, 0, 0, 0, 0, 1, 0, 0, ..., 0, 0, 0, 1]]
	step_tensors = []
	hold_indices = set()   # track active holds; for ssc, a '0' between '2' ... '3' is a hold

	if not sequence:
		return

	num_steps = len(sequence[-1])

	for step in sequence:
		#print(step)
		step_list = [int(symbol) for symbol in step]

		symbol_tensors = []
		for i, symbol in enumerate(step_list):            
			# treat hold starts ('2') as steps ('1'),
			if symbol == SSC_HOLD_SYMBOL:
				hold_indices.add(symbol)
				step_list[i] = SSC_STEP_SYMBOL
			elif i in hold_indices:
				if symbol == SSC_RELEASE_SYMBOL and symbol in hold_indices:
					hold_indices.remove(symbol)
				# treat hold states ('0' between '2'..'3') as holds ('2')
				elif symbol == SSC_OFF_SYMBOL:
					step_list[i] = SSC_HOLD_SYMBOL

			symbol_tensors.append(torch.zeros(SSC_NUM_SYMBOLS).scatter_(0, torch.tensor(symbol), 1))

		# convert symbols -> concatenated one hot encodings
		step_tensors.append(torch.cat(symbol_tensors))

	return torch.cat(step_tensors).view(-1, SSC_NUM_SYMBOLS * num_steps)

@memory.cache
def permute_steps(steps, chart_type, permutation_type, filetype):
	# permutation numbers signify moved location of original step
	#   ex) steps = '10010'
	#       permt = '43210' -> (flip horizontally)
	#    newsteps = '01001'            
	if filetype == 'ucs':
		steps = ucs_to_ssc(steps)

	# replace 'F' ('finish' indicator?) with '0'
	steps = steps.replace('F', '0')
	steps = steps.replace('|', '0')

	if not permutation_type:
		return steps

	permutation = CHART_PERMUTATIONS[chart_type][permutation_type]

	return ''.join([steps[int(permutation[i])] for i in range(len(permutation))])

@memory.cache
def parse_notes(notes, chart_type, permutation_type, filetype):
	# [# frames] - numbers of (10ms) frames for each step
	step_frames = []

	# [# frames] for each frame in ^, whether or not a step was placed or not
	step_placements = []

	# [# frames, # step features] - sequence of non-empty steps with associated frame numbers
	step_sequence = []

	for _, _, time, steps in notes:
		if time < 0:
			continue

		# list containing absolute frame numbers corresponding to each split
		step_frames.append(int(round(time * CHART_FRAME_RATE)))

		# for each frame, 0 = no step, 1 = some step
		step_this_frame = STEP_PATTERNS[filetype].search(steps)
		step_placements.append(1 if step_this_frame else 0)

		# only store non-empty steps in the sequence
		if step_this_frame:
			step_sequence.append(permute_steps(steps, chart_type, permutation_type, filetype))

	return step_frames, step_placements, sequence_to_tensor(step_sequence)

def step_sequence_to_targets(step_sequence):
	"""given a (sequence) of step inputs, return a tensor containing the corresponding vocabulary indices
		in: step_sequence - shape [seq_length, chart_features]
		out: targets - shape [seq_length], values in range [0, vocab size - 1] U [pad_idx]
	"""
	num_arrows = step_sequence.size(1) // NUM_ARROW_STATES
	targets = torch.zeros(step_sequence.size(0), dtype=torch.long)

	for s in range(step_sequence.size(0)):
		idx = 0

		# step_index = SUM(i=0->?)[step[i] * (4^i)] (Base 4 L->R); ? = total arrows
		for i in range(num_arrows):
			start = (i * NUM_ARROW_STATES)
			curr_idx = (step_sequence[s, start:start + NUM_ARROW_STATES] == 1).nonzero(as_tuple=False).flatten()

			idx += curr_idx * (NUM_ARROW_STATES ** i)

		targets[s] = idx

	return targets

def get_step_placement_frames(step_placements, step_frames, batch_size):
	"""Given a sequence of step placements and 10ms frame numbers, return a list of the
		frame numbers of step_frames which have a step (a '1'); Along with first/last frame
	In:
		step_placements: [batch, chart frames] - values in {0,1}, indicating a step or not
		step_frames: [batch, chart frames (variable)] - 10ms audio frame numbers that chart frames occurred at
	Out:
		step_placement_frames: [batch, chart frames with a step (variable)]
	"""	
	step_placement_frames = []
	first_frame = float('inf')
	last_frame = float('-inf')
	for b in range(batch_size):
	    # the indices of step_frames which had step placements
		placement_indices = (step_placements[b] == 1).nonzero(as_tuple=False).flatten().tolist()

		# the subset of step_frames (actual chart frame numbers) that had placements
		step_placement_frames.append([step_frames[b][idx] for idx in placement_indices])

		curr_first_frame = step_frames[b][placement_indices[0]]

		# track the first and last frames with chart non-emtpy step placements
		if curr_first_frame < first_frame:
			first_frame = curr_first_frame

		curr_last_frame = step_frames[b][placement_indices[-1]]

		if curr_last_frame > last_frame:
			last_frame = curr_last_frame

	return step_placement_frames, first_frame, last_frame

class Chart:
	"""A chart object, with associated data. Represents a single example"""

	def __init__(self, chart_attrs, song, filetype, permutation_type=None):
		self.song = song
		self.filetype = filetype

		self.step_artist = chart_attrs['credit'] if 'credit' in chart_attrs else ''
		self.level = int(chart_attrs['meter'])
		self.chart_type = chart_attrs['stepstype']
		assert(self.chart_type in CHART_PERMUTATIONS)

		# concat one-hot encodings of chart_type/level
		chart_type_one_hot = [0 if self.chart_type == 'pump-single' else 1
							  for _ in range(N_CHART_TYPES)]
		level_one_hot = [0] * N_LEVELS
		level_one_hot[self.level - 1] = 1
		self.chart_feats = chart_type_one_hot + level_one_hot

		self.permutation_type = permutation_type

		self.step_frames, self.step_placements, self.step_sequence = parse_notes(
			chart_attrs['notes'], self.chart_type, self.permutation_type, self.filetype)

		self.step_targets = step_sequence_to_targets(self.step_sequence)

	# return tensor of audio feats for this chart
	def get_audio_feats(self):
		return self.song.audio_feats