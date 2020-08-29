## train

This directory contains code defining the models and training process. Implementation
is done using PyTorch.

The models and training procedures are largely based off of the work presented in
[Dance Dance Convolution [DDC] (Donahue et. al, 2017)](https://github.com/chrisdonahue/ddc). In particular, it
presents the task of 'learning to choreograph' as a two step process consisting of
**step placement** and **step selection**.

### Step Placement

**Step placement** determines the particular timesteps at which note(s) in a chart are to occur,
given specified intervals. We follow DDC's standard of using 10ms timesteps. That is, for every 10ms
interval in an audio file, the step placement task must determine whether or not a step should be placed
there. Because level difficulties and other features may also play a role in step placement, this
information is also used to help the models perform this task. The process for step placement is similar to
the one presented in DDC (sections 4.2/4.3 [here](https://arxiv.org/pdf/1703.06891.pdf)). In addition to 
concatenating one-hot representations of chart level, we concatenate one-hot representations of chart type
(single or double) as well, to represent the difference in chart densities between singles and doubles charts.

### Step Selection

**Step selection** determines the particular arrangement of steps at any particular frame with a placement. The
selection models can be trained to either generate singles or doubles charts (but not both). As done 
in DDC, step selection is treated as a language modeling task, where we aim to predict the next step(s), 
given the previous sequence of steps. In our case, the vocabulary of the 'language' consists of the different possible states of the dance pad.

In particular, at any given frame we can classify the state of an arrow on the dance pad in 4 different ways: Off, On, Held, and Released. 
For singles where the dance pad uses 5 arrows, the vocabulary size is 4<sup>5</sup> = 1,024.
For doubles, where all 10 arrows are used, this number grows quite large to 4<sup>10</sup> = 1,048,576.
While using the entire vocabulary space allows every possible combination of arrows, a significant portion of
these arrangements are likely never used (e.g. most arrangements of 5+ arrows at a time). For memory and speed
optimizations, we decide to truncate the vocabulary for doubles primarily to step arrangements of up to 4 arrows at
any one time. We do however allow certain exceptions to be considered if they appear in the training data, 
such as steps from charts like Another Truth D17/18, Hi-Bi D20, Achluoias D26, etc. where more than 4 arrows are 
activated (on, off, or release) at any given time. In this case, these special tokens are appended to the base vocabulary. Altogether this gives a much smaller base vocab size of

<img src="https://latex.codecogs.com/gif.latex?4%5E%7B10%7D%20-%20%5Csum_%7Bi%3D5%7D%5E%7B10%7D%20%7B10%20%5Cchoose%20i%7D%20%5Ccdot%203%5E%7Bi%7D%20%3D%2020%2C686">

We provide two different step selection models. The first is adapted from the LSTM RNN architecture presented in DDC,
section 4.4. We make the addition of taking into account the LSTM outputs of the placement model, using a weighted sum
of these outputs and the previous hidden states to compute a weighted hidden state for the selection RNN. Intuitively,
this design aims to provide the selection model the audio context to help it select a step, as many musical events often
coincide with particular step arrangements (e.g. accents -> jumps, alternating notes -> drills, etc.). 

The second is an Arrow Transformer [WIP], which uses a relative self-attention mechanism. This mechanism enables the model
to more meaningfully capture long-distance dependencies across entire step sequences. In many human-written 
step charts, elements are often repeated, contrasted, extended, and built upon throughout the chart. This type of 
dependency is also present in music itself. The architecture is largely similar to the one presented in [Music Transformer](https://arxiv.org/abs/1809.04281). To provide this model with audio context, we ___

### Examples

Train the CLSTM/RNN placement and selection models:

```bash
# train a model on the dataset 'test_dataset' located in ../data/dataset/subsets/test_dataset.json
# model saved > ./models/{single/double}/test_dataset/
python train_rnns.py --dataset_name=test_dataset

# Train the arrow transformer model, similar as above > ./models/{single/double}/test_dataset/
# can specify --existing_placement=models/already_trained_model/clstm.bin to avoid training
# another placement model from scratch
python train_transformer.py --dataset_name=test_dataset 

# visualize training stats/metrics (requires tensorboard)
tensorboard --logdir=models/single/test_dataset/runs

# To resume training from a checkpoint 
# (use --retrain to start from epoch 0 but still load existing parameters, (e.g. for finetuning))
python train_rnns.py --load_checkpoint='models/single/test_dataset'
```
