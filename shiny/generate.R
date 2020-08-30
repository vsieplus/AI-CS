# use a model to generate a chart with the specified settings

library(rjson)
library(reticulate) 

# change to your python path/conda environment
use_python('/home/vsie/anaconda3/bin/python3.8')
use_condaenv('aics')

generate <- import_from_path('generate', path = file.path('..', 'generate'))

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
        paste(sep = '<br/>', sprintf('Model type: %s', modelDescriptions[[modelSummary[['type']]]]),
              sprintf('Epochs trained: %0.f', modelSummary[['epochs_trained']]),
              sprintf('Batch size: %0.f', modelSummary[['batch_size']]),
              sprintf('Hidden size: %0.f', modelSummary[['hidden_size']]),
              sprintf('Selection hidden weight: %0.2f\n', modelSummary[['selection_hidden_wt']]),
              sprintf('Dataset name: %s', modelSummary[['name']]),
              sprintf('Song types: %s', cat(modelSummary[['song_types']])),
              sprintf('Chart type: %s', modelSummary[['chart_type']]),
              sprintf('Chart permutations: %s', cat(modelSummary[['permutations']])),
              sprintf('Total unique songs: %0.f', modelSummary[['unique_songs']]),
              sprintf('Total audio hours: %0.4f', modelSummary[['audio_hours']]),
              sprintf('Total unique charts: %0.f', modelSummary[['unique_charts']]),
              sprintf('Minimum chart level: %0.f', modelSummary[['min_level']]),
              sprintf('Maximum chart level: %0.f', modelSummary[['max_level']]),
              sprintf('Total chart steps: %0.f', modelSummary[['total_steps']]),
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
generateChart <- function(audioPath, modelPath, level, chartType, title, artist, 
                          bpm, saveFormats, updateProgress = NULL) {
  chartData <-  list('title' = title, 'artist' = artist, 'bpm' = bpm, 'audioPath' = audioPath,
                     'saveFormats' = saveFormats, 'chartType' = paste0('pump-', chartType), 'level' = level)
  
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
    # TODO
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
  
  chartType = modelSummary[['chart_type']]
  inputSize = as.integer(modelSummary[['selection_input_size']])

  if(showUpdates) {
    updateProgress(value = 0.3, detail = 'Generating step placements')
  }
  
  chartPlacements <- generate$generate_placements(placementModel, audioPath, chartType, 
                                                  as.integer(level), inputSize)

  placements = chartPlacements[[1]]
  chartData[['peaks']] = chartPlacements[[2]]
  placementHiddens = chartPlacements[[3]]
  sampleRate = as.integer(chartPlacements[[4]])

  if(showUpdates) {
    updateProgress(value = 0.7, detail = 'Selecting step sequence..')
  }

  chartData[['notes']] <- generate$generate_steps(selectionModel, placements, placementHiddens,
                                                  inputSize, chartType, sampleRate, specialTokens, 
                                                  sampling = 'top-k', k = 25L, p = 0.01)
  if(showUpdates) {
    updateProgress(value = 1.0, detail = 'Complete!')
  }

  chartData
}

# save charts to (temp. path) 'file'
saveCharts <- function(file) {
  if(is.null(chartData)) {
    return(NULL)
  }

  # can access reactive 'chartData' list in here
  chart_df <- chartData()
  
  # use temp directory before zipping file
  origDir = setwd(tempdir())
  on.exit(setwd(origDir))
  
  if(length(chart_df[['saveFormats']]) > 1) {
    saveFormat = 'both'
  } else {
    saveFormat = chart_df[['saveFormats']][1]
  }
  
  generate$save_chart(chart_df[['notes']], chart_df[['chartType']], chart_df[['level']],
                      saveFormat, chart_df[['title']], chart_df[['artist']],
                      chart_df[['audio_path']], chart_df[['name']], '.')
  
  zip(file, list.files('.'))  
}
