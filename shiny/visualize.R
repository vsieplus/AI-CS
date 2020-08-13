# functions to visualize audio/chart properties

library(dplyr)
library(ggplot2)
library(ggimage)
library(av)

library(reticulate) # change to path to your python version
parse <- import_from_path('parse', path = file.path('..', 'data', 'processing'))

samplePath = file.path('www', 'clip.wav')

# return a plot of a spectrogram 
plotSpectrogram <- function(audioPath, startTime = 45.0, endTime = 55.0, saveMP4 = FALSE) {
  fft_data <- av::read_audio_fft(audioPath, start_time = startTime, 
                                 end_time = endTime)
  # save sample clip for output
  if(file.exists(samplePath)) {
    file.remove(samplePath)
  }
  av::av_audio_convert(audioPath, samplePath, start_time = startTime, 
                       total_time = endTime - startTime)
  
  if(saveMP4) {
    av::av_spectrogram_video(samplePath, output = file.path('www', 'clip.mp4'),
                             width = 640, height = 480, res = 144,
                             framerate = 10)
  }
  
  plot(fft_data)
}

# returns a df with two columns, one row per time/step in the chart
getNoteData <- function(chartPath) {
  chartName <- basename(chartPath)
  fileType <- strsplit(chartName, '.', fixed = TRUE)[[1]][-1]
  
  if(file.exists(chartPath)) {
    chartTxt <- gsub('\r', '', readChar(chartPath, file.info(chartPath)$size))
    
    # list of length: # chart frames, each elem has the frame's
    # (beat phase [measure, beat split], abs. beat, time, notes)
    # ex) notes[[100]][[3]] = the time (s) at which the 100th frame occurs
    notes <- parse$ucs_notes_parser(chartTxt)
    
    # extract 3rd/4th value (time/note) of each list element
    # restrict notes to display to start/end time
    times <- unlist(sapply(notes, '[', 3))
    steps <- unlist(sapply(notes, '[', 4))
    
    data.frame(time = times, step = steps, stringsAsFactors = TRUE)
  } else {
    print(sprintf('Chart file %s does not exist', chartPath))
    return(NULL)
  }
}

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
    if(startedHolds != -1) {
      holds[startedHolds] <- TRUE
    }
    
    if(any(holds)) {
      zeroHolds <- intersect(unlist(gregexpr('0', steps[i])), which(holds))
      sapply(zeroHolds, function(idx) substr(steps[i], idx, idx) <<- 'H')
    }    
    
    releasedHolds <- unlist(gregexpr('W', steps[i]))
    if(releasedHolds != -1) {
      holds[releasedHolds] <- FALSE
    }
  }

  # remaining '0's + any other invalid chars -> '.'
  steps <- gsub('0|[^MWHX]', '.', steps)
  steps  
}

# produce a plot of step chart section
# noteData should be an m by 2 df, with observations of time/step
plotChartSection <- function(noteData, format = 'ucs', startTime = 45.0, endTime = 55.0) {
  if(format == 'ssc') {
    # convert notes to ucs
    # df$step <- lapply(df$step, convertToUCS)
  }
  
  noteData <- filter(noteData, time >= startTime & time <= endTime)
  
  # y axis time (going down)
#  ggplot2::ggplot(noteData, aes(y = time, x = step)) +
    
}

# color based on step position (.X.X. -> red, X...X -> blue, ..X.. -> yellow)
notePositions <- list(top = '[1368]', center = '[27]', bottom = '[0459]')
noteColors <- c('r' = 'red', 'y' = 'yellow', 'b' = 'blue', 'ryb' = 'black',
                'ry' = 'orange', 'rb' = 'purple', 'yb' = 'green')

# add icons to axis labels

# TODO try stacked bar chart, once col. for each step, sep. steps, holds
# produce a plot of the entire step chart's distribution
# same input type as above
plotChartDistribution <- function(noteData, format = 'ucs') {
  if(format == 'ssc') {
    noteData$step <- convertToUCS(noteData$step)
  }
  
  # filter empty steps + extract note positions
  noteData <- filter(noteData, step != '.....')
  noteData$positions <- lapply(noteData$step, function(s) {
    unlist(gregexpr('[XMHW]', s)) - 1 
  })
  
  # map colors to steps; 2 or 7: center; otherwise
  # (1368) ~ top steps: red), (0459 ~ bottom: blue)
  noteData$color <- factor(sapply(noteData$positions, function(indices) {
    color <- ''
    if(length(grep(notePositions[['top']], indices)) > 0) {
      color <- paste0(color, 'r') 
    }
    
    if(length(grep(notePositions[['center']], indices)) > 0) {
      color <- paste0(color, 'y') 
    }
    
    if(length(grep(notePositions[['bottom']], indices)) > 0) {
      color <- paste0(color, 'b') 
    }
    color
  }))
  
  # sort/filter freqs
  freqs <- summary(noteData$step)
  noteData$freqs <- lapply(noteData$step, function(s) freqs[names(freqs) == s])
  noteData$step <- reorder(noteData$step, noteData$step, FUN=length)
  
  noteData <- filter(noteData, freqs > 10)
  
  maxFreq <- max(freqs)
  freqSteps <- seq(0, (maxFreq %/% 50) * 50 + 50, 25)
  
  ggplot2::ggplot(noteData, aes(x = step, fill = color)) +
    geom_bar(width = 0.75, color = 'black', alpha = 0.8) + 
    theme(legend.position = 'none') +
    labs(x = 'Step type', y = 'Frequency', 
         title = 'Distribution of chart steps (freq. > 10)') +
    coord_flip() +
    scale_y_continuous(expand = c(0,0), breaks = freqSteps,
                       limits = c(0, max(freqSteps))) +
    scale_fill_manual(values = noteColors, aesthetics = c('color', 'fill'))
}

