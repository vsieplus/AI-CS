# utility classes for loading step charts

import json
import math
import os
import re

import torch
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from pathlib import Path
from joblib import Memory

from extract_audio_feats import extract_audio_feats, load_audio
from hyper import (HOP_LENGTH, PAD_IDX, SEED, N_CHART_TYPES, N_LEVELS, CHART_FRAME_RATE, 
				   NUM_ARROW_STATES, MAX_ACTIVE_ARROWS, SELECTION_VOCAB_SIZES)
from util import convert_chartframe_to_melframe

# cache dataset tensors/other values https://discuss.pytorch.org/t/cache-datasets-pre-processing/1062/8
ABS_PATH = str(Path(__file__).parent.absolute())
CACHE_DIR = os.path.join(ABS_PATH, '.dataset_cache/')

memory = Memory(CACHE_DIR, verbose=0, compress=True)
# to reset cache: memory.clear(warn=False)

# add caching to extract_audio_feats
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

	first_frame = 100000000
	last_frame = -1

	placement_targets = []

	step_sequence = []
	step_targets = []

	# list of lengths of examples, for packing
	audio_lengths = []
	step_sequence_lengths = []

	for chart in batch:
		# transpose channels/timestep so timestep comes first for pad_sequence()
		audio_feats.append(chart.get_audio_feats().transpose(0, 1))
		chart_feats.append(torch.tensor(chart.chart_feats).unsqueeze(0))
		placement_targets.append(chart.placement_targets)

		levels.append(chart.level)

		# already tensors
		step_sequence.append(chart.step_sequence)
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

class StepchartDataset(Dataset):
	"""Dataset of step charts"""
	def __init__(self, dataset_json):
		"""dataset_json: path to json file with dataset metadata"""
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

		self.permutations = metadata['permutations'] + ['']
		
		self.splits = metadata['splits']

		self.n_unique_songs = 0

		# load the actual training samples
		self.load_charts(metadata['json_fps'])
		self.print_summary()
		self.compute_stats()

		# gather 'outlier' steps (to add to vocab)
		self.outliers = set()
		for chart in self.charts:
			self.outliers.add(outlier for outlier in chart.outliers)

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

	def compute_stats(self):
		self.n_unique_charts = self.__len__() // len(self.permutations)
		self.n_steps = sum([chart.n_steps for chart in self.charts])
		self.n_audio_hours = sum([song.n_minutes for song in self.songs.values()]) / 60

		if not self.step_artists:
			self.step_artists = set()
			for chart in self.charts:
				self.step_artists.add(chart.step_artist)

		steps_per_second = [chart.steps_per_second for chart in self.charts]
		self.avg_steps_per_second = sum(steps_per_second) / len(steps_per_second)

	# filter/load charts
	def load_charts(self, json_fps):
		print('Loading charts...')
		self.songs = {}
		self.charts = []
		for fp in tqdm(json_fps):
			with open(fp, 'r') as f:
				attrs = json.loads(f.read())
			
			# check attrs for each chart to see if we should add it to the dataset
			chart_indices = self.filter_charts(attrs)

			if not chart_indices:
				continue

			# .ssc (may contain multiple charts/same song) or .ucs (always 1 chart)    
			orig_filetype = attrs['chart_fp'].split('.')[-1]

			# create new song if needed
			song_path = attrs['music_fp']
			if song_path not in self.songs:
				self.songs[song_path] = Song(song_path, attrs['title'], attrs['artist'],
											 attrs['genre'], attrs['songtype'])
				self.n_unique_songs += 1

			for chart_idx in chart_indices:
				for permutation in self.permutations:
					self.charts.append(Chart(attrs['charts'][chart_idx], self.songs[song_path],
											 orig_filetype, permutation))
		
		print('Done loading!')
	
	def filter_charts(self, attrs):
		""" determine which charts in the given attrs belongs in this dataset
			return list of indices, w/length between 0 <= ... <= len(attrs['charts'])
	   	"""
		chart_indices = []

		if not self.songtypes or self.songtypes and attrs['songtype'] in self.songtypes:
			for i, chart_attrs in enumerate(attrs['charts']):
				if chart_attrs['stepstype'] != self.chart_type:
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

		# shape [3, ?, 80]
		self.audio_feats, self.sample_rate = extract_audio_feats(self.audio_fp)

		# secs = (hop * melframe) / sample_rate
		self.n_minutes = (HOP_LENGTH * self.audio_feats.size(1) / self.sample_rate) / 60

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
	
	if not permutation_type:
		return steps

	permutation = CHART_PERMUTATIONS[chart_type][permutation_type]

	return ''.join([steps[int(permutation[i])] for i in range(len(permutation))])

@memory.cache
def parse_notes(notes, chart_type, permutation_type, filetype):
	# [# frames] track which (10 ms) chart frames have steps
	step_placement_frames = []

	# [# frames, # step features] - sequence of non-empty steps with associated frame numbers
	step_sequence = []

	for i, (_, _, time, steps) in enumerate(notes):
		if time < 0:
			continue

		# for each frame, track absolute frame number
		step_this_frame = STEP_PATTERNS[filetype].search(steps)
		if step_this_frame:
			step_placement_frames.append(int(round(time * CHART_FRAME_RATE)))

		# only store non-empty steps in the sequence
		if step_this_frame:
			step_sequence.append(permute_steps(steps, chart_type, permutation_type, filetype))

	return step_placement_frames, step_sequence
	
def step_sequence_to_targets(step_input, step_sequence, chart_type):
	"""
	given a (sequence) of step inputs, return a tensor containing the corresponding vocabulary indices
		in: step_input - shape [seq length, chart_features], tensor representations of (nonempty) steps
			step_sequence - shape [seq length] - original string representation of step sequence
		out: targets - shape [seq length], values in range [0, vocab size - 1]
	"""
	num_arrows = len(step_sequence[0])
	outliers = {}	# for doubles charts, consider steps with 5+ activated arrows as outliers
					# dict 'index -> 

	targets = torch.zeros(step_input.size(0), dtype=torch.long)

	for s in range(step_input.size(0)):
		# index in order of: num active arrows [1, ..., num_arrows - 1] (ignore 0)
		# 				  -> arrow indices [0, ..., num_arrows - 1] x num_active_arrows
		#				  -> arrow states [1, ..., num_arrow_states - 1] (ignore off)

		# step_index =  SUM_{k=0->num_active - 1} (num_arrows choose k) * 3 ^ k +
		#				SUM_{i' < i} ... SUM_{j' < j} 3 ^ num_active +
		#				SUM_{x1, .., x_num_active} (x_i - 1) * 3 ^ i (base 3 R -> L)
		#	[i' = first index L->R, i = true first index, j', j = zth index, z = num active arrows]
		#				
		idx = 0

		active_arrow_states = []
		active_arrow_indices = []
		for i in range(step_input.size(1) // NUM_ARROW_STATES):
			start = i * NUM_ARROW_STATES
			arrow_state = (step_input[s, start:start + NUM_ARROW_STATES] == 1).nonzero(as_tuple=False).flatten()

			if arrow_state != 0:
				active_arrow_indices.append(i)
				active_arrow_states.append(arrow_state)

		num_active_arrows = len(active_arrow_indices)

		# if num active exceed maximum, add as an outlier
		if num_active_arrows > MAX_ACTIVE_ARROWS[chart_type]:
			outliers.append(step)
			targets[s] = SELECTION_VOCAB_SIZES[chart_type]
			## TODO figure out target/vocab size modification for outliers
			continue

		for num_active in range(num_active_arrows):
			idx += math.comb(num_arrows, num_active) * (3 ** num_active)

		first_possible_idx = 0
		for arrow_idx in active_arrow_indices:
			if arrow_idx > first_possible_idx:
				for j in range(arrow_idx):
					idx += 3 ** num_active_arrows * (num_arrows - j - 1)
			
			first_possible_idx = arrow_idx + 1
		
		# base 3 R -> L
		active_arrow_states.reverse()
		for a, state in enumerate(active_arrow_states):
			idx += (state - 1) * (3 ** a)

		targets[s] = idx

	return targets, outliers

def placement_frames_to_targets(placement_frames, audio_length, sample_rate):
	# get chart frame numbers of step placements, [batch, placements (variable)]
	placement_target = torch.zeros(audio_length, dtype=torch.long)

	mel_placement_frames = []
	for f in placement_frames:
		melframe = convert_chartframe_to_melframe(f, sample_rate)

		# ignore steps that come after the song has ended
		if melframe < audio_length:
			mel_placement_frames.append(melframe)

	mel_placement_frames = torch.tensor(mel_placement_frames, dtype=torch.long)

	# and set target to '1' at corresponding melframes
	placement_target[mel_placement_frames] = 1

	first_frame = mel_placement_frames[0].item()
	last_frame = mel_placement_frames[-1].item()

	return placement_target, first_frame, last_frame

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
		chart_type_one_hot = [0] * N_CHART_TYPES
		if self.chart_type == 'pump-single':
			chart_type_one_hot[0] = 1
		else:
			chart_type_one_hot[1] = 1
		level_one_hot = [0] * N_LEVELS
		level_one_hot[self.level - 1] = 1
		self.chart_feats = chart_type_one_hot + level_one_hot

		self.permutation_type = permutation_type

		(step_placement_frames, step_sequence) = parse_notes(chart_attrs['notes'], self.chart_type,
														     self.permutation_type, self.filetype)

		(self.placement_targets,
		 self.first_frame, 
		 self.last_frame) = placement_frames_to_targets(step_placement_frames,
		  			   								    self.song.audio_feats.size(1),
													  	self.song.sample_rate)

		self.step_sequence = sequence_to_tensor(step_sequence)
		self.step_targets, self.outliers = step_sequence_to_targets(self.step_sequence, step_sequence, self.chart_type)

		# ignore steps in sequence after song has ended
		# (make sure length matches num of placement targets)
		self.step_sequence = self.step_sequence[:self.placement_targets.sum()]
		self.step_targets = self.step_targets[:self.placement_targets.sum()]

		self.n_steps = self.step_sequence.size(0)
		self.steps_per_second = self.n_steps / (self.song.n_minutes * 60)

	# return tensor of audio feats for this chart
	def get_audio_feats(self):
		return self.song.audio_feats