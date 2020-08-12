# use a model to generate a chart with the specified settings

library(reticulate) # https://rstudio.github.io/reticulate/
library(rjson)

CHART_LEVELS <- list('single' = 26, 'double' = 28)

# get a list of directories in root/, and their respective paths
getModelDirs <- function(root) {
  dirnames <- list.dirs(root, full.names  =  FALSE)
  dirpaths <- list.dirs(root, full.names  =  TRUE)
  names(dirpaths) = dirnames
  
  # exclude the root
  dirpaths[2:length(dirnames)]
}

# you can change modelsDir (relative path) to another folder containing
# your models when running locally; should have structure
# models/
#   single/
#     ...
#   double/
#     model1/
#       model1.tar (the model itself saved from pytorch)
#       summary.json (produced at end of training)
#     ...
modelsDir <- 'models'
modelTypes <- c('single', 'double')
modelsList <- lapply(sapply(modelTypes, 
                            function(x) file.path(modelsDir, x)), 
                     getModelDirs)

# retrieve text output containing a summary of the given model
getModelSummary <- function(modelPath) {
  model_json_path <- file.path(modelPath, 'summary.json')
  
  if(dir.exists(modelPath)) {
    if(file.exists(model_json_path)) {
      # TODO implement model summary.json format from training
      #model_summary <- rjson::fromJSON(file = model_json_path)
      modelName <- 'hi'
      sprintf('Model Name: %s', modelName)
      
      # summaryString
    } else {
      sprintf('summar.json not found in %s', modelPath)
    }
  } else {
    sprintf('Model directory %s not found', modelPath)
  }
}

# generate a chart
generate_chart <- function(modelPath, chart_type, level) {
  
}
