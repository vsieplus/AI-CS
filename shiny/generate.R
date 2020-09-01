# use a model to generate a chart with the specified settings

library(rjson)
library(reticulate) 

# change to your python path/conda environment
use_python('/home/vsie/anaconda3/bin/python3.8')
use_condaenv('aics')

generate <- import_from_path('generate', path = file.path('..', 'generate'))
hyper <- import_from_path('hyper', path = file.path('..', 'train'))

CHART_LEVELS <- list('single' = 26, 'double' = 28)

modelDescriptions = list('rnns' = 'C-LSTM Placement Model + Weighted LSTM RNN Selection Model')

# retrieve text output containing a summary of the given model
getModelSummary <- function(modelPath, as_str) {
  modelJsonPath <- file.path(modelPath, 'summary.json')
  
  if(dir.exists(modelPath)) {
    if(file.exists(modelJsonPath)) {
      # model summary.json from training
      modelSummary <- rjson::fromJSON(file = modelJsonPath)

      if(as_str) {
        paste(sep = '<br/>', sprintf('Model description: %s', modelDescriptions[[modelSummary[['type']]]]),
              sprintf('Vocabulary size: %s', format(modelSummary[['vocab_size']], big.mark = ',', trim = T)),
              sprintf('Epochs trained: %0.f', modelSummary[['epochs_trained']]),
              sprintf('Batch size: %0.f', modelSummary[['batch_size']]),
              sprintf('Hidden size: %0.f', modelSummary[['hidden_size']]),
              sprintf('Selection hidden weight: %0.2f<br/>', modelSummary[['selection_hidden_wt']]),
              sprintf('Dataset name: %s<br/>', modelSummary[['name']]),
              sprintf('Song types: %s', cat(modelSummary[['song_types']])),
              sprintf('Total unique songs: %0.f', modelSummary[['unique_songs']]),
              sprintf('Total audio hours: %0.4f<br/>', modelSummary[['audio_hours']]),
              sprintf('Chart type: %s', modelSummary[['chart_type']]),
              sprintf('Chart permutations: %s', cat(modelSummary[['permutations']])),
              sprintf('Total unique charts: %s', format(modelSummary[['unique_charts']], big.mark = ',', trim = T)),
              sprintf('Minimum chart level: %0.f', modelSummary[['min_level']]),
              sprintf('Maximum chart level: %0.f', modelSummary[['max_level']]),
              sprintf('Total chart steps: %s', format(modelSummary[['total_steps']], big.mark = ',', trim = T)),
              sprintf('Avg. steps per second: %0.4f', modelSummary[['avg_steps_per_second']]))
      } else {
        modelSummary
      }
    } else {
      sprintf('summary.json not found in %s', modelPath)
    }
  } else {
    sprintf('Model directory %s not found', modelPath)
  }
}

# generate a chart; return a list with the generated notes + chart metadata
generateChart <- function(audioPath, modelPath, level, chartType, title, artist, bpm, saveFormats,
                          sampleStrat, topkK, toppP, beamSize, updateProgress = NULL) {
  if(sampleStrat == 'top-k') {
    sampleDescription <- sprintf("Top-k sampling with k = %0.f", topkK)
  } else if(sampleStrat == 'top-p') {
    sampleDescription <- sprintf("Top-p sampling with p = %0.f", toppP)
  } else if(sampleStrat == 'beam-search') {
    sampleDescription <- sprintf("Beam-search with beam size b = %0.f", beamSize)
  } else {
    sampleDescription <- sampleStrat
  }

  chartData <-  list('title' = title, 'artist' = artist, 'bpm' = bpm, 'audioPath' = audioPath,
                     'saveFormats' = saveFormats, 'chartType' = chartType, 'level' = as.integer(level),
                     'sampleStrategy' = sampleDescription)
  
  showUpdates = is.function(updateProgress)

  if(showUpdates) {
    updateProgress(value = 0, 'Loading model summary')
  }
  
  modelSummary <- getModelSummary(modelPath, as_str = FALSE)
  
  chartData[['name']] = modelSummary[['name']]
  
  if(showUpdates) {
    updateProgress(value = 0.1, detail = 'Loding models')
  }

  # convert R numeric values to ints
  if(modelSummary[['type']] == 'rnns') {
    modelIntFeatures <- c('placement_channels', 'placement_kernels', 'placement_filters',
                          'placement_pool_kernel', 'placement_pool_stride', 'placement_lstm_layers',
                          'placement_input_size', 'hidden_size', 'selection_lstm_layers',
                          'selection_input_size', 'vocab_size')
  } else {
    # TODO (transformer)
  }

  sapply(modelIntFeatures, function(s) {
    if(is.list(modelSummary[[s]])) {
      modelSummary[[s]] <<- lapply(modelSummary[[s]], as.integer)
    } else {
      modelSummary[[s]] <<- sapply(modelSummary[[s]], as.integer)
    }
  })

  genConfig <- generate$get_gen_config(modelSummary, modelPath)

  placementModel = genConfig[[1]]
  selectionModel = genConfig[[2]]
  specialTokens = genConfig[[3]]
  thresholds = genConfig[[4]]
  
  chartType = modelSummary[['chart_type']]
  vocabSize = modelSummary[['vocab_size']]
  inputSize = modelSummary[['selection_input_size']]

  if(showUpdates) {
    updateProgress(value = 0.3, detail = 'Generating step placements')
  }
  
  chartData[['threshold']] <- hyper$PLACEMENT_THRESHOLDS[as.integer(level - 1)]

  chartPlacements <- generate$generate_placements(placementModel, audioPath, chartType, 
                                                  as.integer(level), thresholds, inputSize)

  placements = chartPlacements[[1]]
  chartData[['peaks']] = chartPlacements[[2]]
  placementHiddens = chartPlacements[[3]]
  sampleRate = as.integer(chartPlacements[[4]])

  if(showUpdates) {
    updateProgress(value = 0.7, detail = 'Selecting step sequence')
    Sys.sleep(1)
  }

  generatedSteps <- generate$generate_steps(selectionModel, placements, placementHiddens, vocabSize,
                                            inputSize, chartType, sampleRate, specialTokens, 
                                            sampling = sampleStrat, k = topkK, p = toppP, b = beamSize)

  chartData[['notes']] <- generatedSteps
  chartData[['notes_df']] <- data.frame(time = sapply(generatedSteps, function(x) x[[1]]),
                                        step = sapply(generatedSteps, function(x) x[[2]]))
  
  chartData[['first_step_time']] <- head(chartData[['notes_df']][['time']], 1)
  chartData[['last_step_time']] <- tail(chartData[['notes_df']][['time']], 1)
  chartData[['num_placements']] <- length(chartData[['notes']])
  chartData[['steps_per_second']] <- chartData[['num_placements']] / chartData[['last_step_time']]
  
  if(showUpdates) {
    updateProgress(value = 1.0, detail = 'Complete!')
    Sys.sleep(2)
  }

  chartData
}

getGenerationSummary <- function(chartData) {
  chartTitle <- paste0(chartData[['title']], ' ', toupper(substr(chartData[['chartType']], 6, 6)), chartData[['level']])
  paste(sep = '<br/>', 
        sprintf('Chart generated: %s', chartTitle),
        sprintf('Sampling strategy: %s', chartData[['sampleStrategy']]),
        sprintf('Total step placements: %0.f', chartData[['num_placements']]),
        sprintf('Avg. steps per second: %0.4f', chartData[['steps_per_second']]),
        sprintf('First step : %0.4f s', chartData[['first_step_time']]),
        sprintf('Last step: %0.4f s', chartData[['last_step_time']]))
}
