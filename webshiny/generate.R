# use a model to generate a chart with the specified settings

library(reticulate) # https://rstudio.github.io/reticulate/
library(rjson)

# you can change this (relative path) to whatever folder you have your models in
# if running locally
MODELS_DIR <- 'models'

get_model_summary <- function(model_name) {
  model_path <- file.path(MODELS_DIR, model_name)
  model_json_path <- file.path(model_path, 'summary.json')
  
  if(dir.exists(model_path) && file.exists(model_json_path)) {
    model_summary <- fromJSON(file=model_json_path)
    sprintf('Model Name: %s', model_name)
  } else {
    sprintf('Model directory %s not found or missing summary.json', model_path)
  }
}
