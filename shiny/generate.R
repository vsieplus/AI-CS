# use a model to generate a chart with the specified settings

library(rjson)
library(reticulate) 

# change to your python path/conda environment
use_python('/home/vsie/anaconda3/bin/python3.8')
use_condaenv('aics')

generate <- import_from_path('generate', path = file.path('..', 'generate'))

CHART_LEVELS <- list('single' = 26, 'double' = 28)

# retrieve text output containing a summary of the given model
getModelSummary <- function(modelPath, as_str) {
  modelJsonPath <- file.path(modelPath, 'summary.json')
  
  if(dir.exists(modelPath)) {
    if(file.exists(modelJsonPath)) {
      # model summary.json from training
      modelSummary <- rjson::fromJSON(file = modelJsonPath)

      if(as_str) {
        paste(sep = '<br/>', sprintf('Model type: %s', modelSummary[['type']]),
              sprintf('Epochs trained: %0.f', modelSummary[['epochs_trained']]),
              sprintf('Batch size: %0.f', modelSummary[['batch_size']]),
              sprintf('Hidden size: %0.f', modelSummary[['hidden_size']]),
              sprintf('Selection hidden weight: %0.2f', modelSummary[['selection_hidden_wt']]),
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
generateChart <- function(audioPath, modelPath, level, chartType, title, artist, bpm, saveFormats) {
  chartData <-  list('title' = title, 'artist' = artist, 'bpm' = bpm, 'audioPath': audioPath,
                    'saveFormats' = saveFormats, 'chartType' = chartType, 'level' = level)
  
  modelSummary <- getModelSummary(modelPath, as_str = FALSE)
  
  chartData[['name']] = modelSummary[['name']]
  
  genConfig <- generate$get_cen_config(modelSummary, modelPath, device)
  
  placementModel = genConfig[[1]]
  selectionModel = genConfig[[2]]
  specialTokens = genConfig[[3]]
  
  chartType = modelSummary[['chart_type']]
  inputSize = modelSummary[['selection_input_size']]

  chartGen <- generate$generate_chart(placementModel, selectionModel, audioPath, chartType,
                                      level, inputSize, specialTokens, sampling = 'top-k',
                                      k = 20, p = 0.01, device)
  chartData[['notes']] = chartGen[[0]]
  chartData[['peaks']] = chartGen[[1]]
}

# save charts to (temp. path) 'file'
saveCharts <- function(file) {
  # can access reactive 'chartData' list in here
  if(is.null(chartData)) {
    return(NULL)
  }
  
  # use temp directory before zipping file
  origDir = setwd(tempdir())
  on.exit(setwd(origDir))
  
  if(length(chartData()[['saveFormats']]) > 1) {
    saveFormat = 'both'
  } else {
    saveFormat = chartData()[['saveFormats']][1]
  }
  
  generate$save_chart(chartData()[['notes']], chartData()[['chartType']], chartData()[['level']],
                      saveFormat, chartData()[['title']], chartData()[['artist']],
                      chartData()[['audio_path']], chartData()[['name']], '.')
  
  zip(file, list.files('.'))  
}
