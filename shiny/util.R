# convert an sequence of ssc steps to UCS notation
convertToUCS <- function(steps) {
  # '1' -> 'X' (step), '2' -> 'M' (start hold), '3' -> 'W' (release hold)
  steps <- gsub('1', 'X', steps)
  steps <- gsub('2', 'M', steps)
  steps <- gsub('3', 'W', steps)
  
  # replace '0's between M ... W -> H (hold)
  holds <- rep(FALSE, nchar(steps[1]))
  for(i in seq(length(steps))) {
    startedHolds <- unlist(gregexpr('M', steps[i]))
    if(length(startedHolds) != 1 || startedHolds != -1) {
      holds[startedHolds] <- TRUE
    }
    
    if(any(holds)) {
      zeroHolds <- intersect(unlist(gregexpr('0', steps[i])), which(holds))
      sapply(zeroHolds, function(idx) substr(steps[i], idx, idx) <<- 'H')
    }    
    
    releasedHolds <- unlist(gregexpr('W', steps[i]))
    if(length(releasedHolds) != 1 || releasedHolds != -1) {
      holds[releasedHolds] <- FALSE
    }
  }

  # remaining '0's + any other invalid chars -> '.'
  steps <- gsub('0|[^MWHX]', '.', steps)
  steps  
}

# get a list of directories in root/, and their respective paths
getModelDirs <- function(root) {
  dirnames <- list.dirs(root, full.names  =  FALSE, recursive = FALSE)
  dirpaths <- list.dirs(root, full.names  =  TRUE, recursive = FALSE)
  names(dirpaths) = dirnames
  
  dirpaths
}

# you can change modelsDir (relative path) to another folder containing
# your models when running locally; should have structure
# models/
#   single/
#     ...
#   double/
#     model1/
#       model.bin (the model file(s) itself saved from training)
#       summary.json (produced at end of training)
#     ...
modelsDir <- file.path('..', 'train', 'models')
#modelsDir <- 'models'
modelTypes <- c('single', 'double')
modelsList <- lapply(sapply(modelTypes, function(x) file.path(modelsDir, x)), getModelDirs)
