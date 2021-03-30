# functions to visualize audio/chart properties
library(dplyr)
library(ggplot2)
library(grid)
library(extrafont)
library(magick)
library(reshape2)
library(av)

## Chart Processing (util functions) ####################################
library(reticulate)
use_python('/home/ryan/anaconda3/bin/python3.8')
use_condaenv('aics')
parse <- reticulate::import_from_path('parse', path = file.path('..', 'data', 'processing'))

source('util.R', local = TRUE)

# loadfonts(device = 'win') [run this the first time loading extrafont]

samplePath = file.path('www', 'clip.wav')

## AUDIO VISUALIZATION ##############################

# return a plot of a spectrogram 
plotSpectrogram <- function(audioPath, startTime = 15.0, endTime = 30.0, saveMP4 = FALSE) {
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

# find the indices of steps and holds
addStepsAndHolds <- function(noteData, fileType = 'ucs') {
  if(fileType == 'ssc') {
    noteData$step <- convertToUCS(noteData$step)
  }
  
  # values will be '-2' if no corresponding arrows found
  noteData$stepIndices <- lapply(noteData$step, function(s) {
    unlist(gregexpr('[X]', s)) - 1
  })
  
  noteData$holdIndices <- lapply(noteData$step, function(s) {
    unlist(gregexpr('[MHW]', s)) - 1
  })
  
  noteData
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
    if(fileType == 'ucs') {
      notes <- parse$ucs_notes_parser(chartTxt)
    } else if(fileType == 'ssc') {
      notes <- parse$parse_ssc_txt(chartTxt)
      
      # extract 3rd/4th value (time/note) of each list element
      # restrict notes to display to start/end time
      charts <- notes[['charts']]
      
      # ask user to choose which chart in the ssc file if > 1
      if(length(charts) > 1) {
        chartIdx <- menu(sapply(charts, function(c) {
            paste0(strsplit(c[['stepstype']], '-')[[1]][-1], c[['meter']])
          }), title = "Choose a chart")
      } else {
        chartIdx <- 1
      }
      
      notes <- charts[[chartIdx]][['notes']]
    }
    
    times <- unlist(sapply(notes, '[', 3))
    steps <- unlist(sapply(notes, '[', 4))
    
    noteData <- data.frame(time = times, step = steps, stringsAsFactors = TRUE)
    addStepsAndHolds(noteData, fileType)
  } else {
    print(sprintf('Chart file %s does not exist', chartPath))
    return(NULL)
  }
}

#################################################################
## CHART/Model VISUALIZATION ####################################

chartTheme <- theme(text = element_text(family = 'Calibri', color = 'white', size = 12),
                    axis.text.y = element_text(color = 'white'),
                    plot.title = element_text(size = 16),
                    plot.background = element_rect(fill = '#131711', color = 'lightblue'),
                    panel.background = element_rect(fill = 'black', color = 'lightblue'),
                    panel.grid.major.y = element_line(color='gray50'),
                    panel.grid.major.x = element_blank(),
                    panel.grid.minor.y = element_line(color = 'gray30'),
                    panel.grid.minor = element_blank())

# plot model peak picking scores for the given range
plotPeakPicking <- function(modelPeaks, threshold, startTime = 15.0, endTime = 30.0) {
  # modelPeaks is a list of lists, each sublist with 2 elems: time + peak score
  peaks <- data.frame(time = sapply(modelPeaks, function(x) x[[1]]),
                      prob = sapply(modelPeaks, function(x) x[[2]]))
  peaks <- filter(peaks, time >= startTime & time <= endTime)
  
  ggplot(peaks, aes(x = time, y = prob)) +
    geom_line(color='white') + 
    geom_hline(yintercept = threshold, color = 'white', size = 1) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1), expand = c(0,.01)) +
    scale_x_continuous(limits = c(startTime, endTime), breaks = seq(startTime, endTime, 1), expand = c(0,0)) +
    labs(title = 'Model placement confidence', x = 'Time (s)', y = 'Probability') +
    theme(axis.text.x = element_text(color = 'white')) +
    chartTheme
}

notePositions <- list(top = '[1368]', center = '[27]', bottom = '[0459]')

# arrow icons to plot
notePaths <- rep(c('www/piudl.png', 'www/piuul.png', 'www/piuc.png',
                   'www/piuur.png', 'www/piudr.png'), 2)
names(notePaths) = seq(1, 10)
noteIcons <- sapply(notePaths, image_read)
noteColors <- c('#c3e9d0', '#8da6e2', '#6184d8', '#1e1014')

## Image axis START ####
# adapted from https://gist.github.com/jonocarroll/2f9490f1f5e7c82ef8b791a4b91fc9ca
icon_axis <- function(arrows, angle) {
  structure(
    list(img=noteIcons[1:arrows], angle=angle),
         class = c("element_custom", "element_blank", "element_text", "element")
  )
}

element_grob.element_custom <- function(element, x, ...)  {
  stopifnot(length(x) == length(element$img))
  tag <- names(element$img)

  # add vertical padding to leave space
  g1 <- textGrob(label = '', x=x, rot = element$angle, vjust=0.6)
  g2 <- mapply(rasterGrob, x=x, image=element$img[tag], 
              MoreArgs=list(vjust=0.5, interpolate=FALSE,
                            height=unit(7 ,"mm")),
              SIMPLIFY=FALSE)
  
  gTree(children=do.call(gList, c(g2, list(g1))), cl="custom_axis")
}

# icon spacing
grobHeight.custom_axis = heightDetails.custom_axis = function(x, ...) {
  unit(12, "mm")
}

## END #####

# produce a horizontal plot of step chart section (i.e. a still-shot of a stepchart section)
# noteData should be an m x 2 df, with observations of time/step
plotChartSection <- function(noteData, startTime = 15.0, endTime = 30.0, epsilon = 1e-5) {
  num_arrows <- nchar(as.character(noteData[1, 'step']))
  noteData <- filter(noteData, time >= startTime & time <= endTime)
  
  arrows <- seq(0, num_arrows - 1)
  stepPositions <- lapply(arrows, function(a) {
    sapply(noteData$stepIndices, function(s) a %in% s)
  })
  
  holdPositions <- lapply(arrows, function(a) {
    sapply(noteData$holdIndices, function(s) a %in% s)
  })

  names(stepPositions) <- paste0(arrows, '-step')
  names(holdPositions) <- paste0(arrows, '-hold')

  noteData <- cbind(noteData, stepPositions)
  
  # for each timestep, find the list of arrow ids [0-num_arrows-1] with a note
  timeNotes <- lapply(noteData$time, function(time) {
    which(sapply(names(stepPositions), function(a) {
      noteData[abs(noteData$time - time) < epsilon, a]
    }))
  })

  names(timeNotes) = round(noteData$time,3)
  timeNotes <- timeNotes[which(sapply(timeNotes, function(t) length(t) > 0))]
  
  # add one data point for each time/index pair
  noteTimes <- data.frame(time = sapply(names(timeNotes), function(t) {
    rep(as.numeric(t), length(timeNotes[[t]]))
  }), notes = unlist(timeNotes))
  
  # time vs notes
  ggplot2::ggplot(noteTimes, aes(y = time, x = notes)) +
    geom_point(color = 'white') + 
    scale_x_continuous(limits = c(0, length(arrows) - 1), breaks = seq(0, length(arrows) - 1, 1)) +
    scale_y_continuous(trans = 'reverse') +
    theme(axis.text.x = icon_axis(arrows = num_arrows, angle = 0),
          axis.title.x = element_blank(),
          axis.text.y = element_text(color = 'white')) +
    chartTheme
}

# count arrow types from note data
# filter_fn should extract subset of note types
# return total size of that subset
filterNotes <- function(arrows, filter_fn) {
  sapply(arrows, filter_fn)
}

getArrowDistribution <- function(noteData, num_arrows) {
  # tabulate no. of steps per arrow, broken down by type
  arrows_seq <- seq(0, num_arrows - 1)
  arrows <- data.frame(arrow = arrows_seq)

  arrows$step <- filterNotes(arrows$arrow, function(a) { 
    sum(unlist(sapply(noteData$stepIndices, function(p) {
      length(p) == 1 && a == p[[1]]
    })))
  })
  
  arrows$jumps <- filterNotes(arrows$arrow, function(a) {
    sum(sapply(noteData$stepIndices, function(s) length(s) == 2 & a %in% s) & 
        sapply(noteData$holdIndices, function(h) all(h == -2)))
  })
  
  arrows$brackets <- filterNotes(arrows$arrow, function(a) {
    sum(sapply(noteData$stepIndices, function(s) length(s) >= 3 & a %in% s) | 
       (sapply(noteData$holdIndices, function(h) all(h != -2)) & 
        sapply(noteData$stepIndices, function(s) {
        length(s) == 2 & a %in% s 
       })))
  })
  
  arrows$holds <- sapply(arrows$arrow, function(a) { 
    sum(sapply(noteData$holdIndices, function(p) a %in% p))
  })
  
  arrows
}

# Plot distribution of chart steps
plotChartDistribution <- function(noteData, chartTitle = '') {
  # filter empty steps + extract note positions
  num_arrows <- nchar(as.character(noteData[1, 'step']))
  noteData <- filter(noteData, step != strrep('.', num_arrows))
  noteData <- addStepsAndHolds(noteData)
  
  arrows <- getArrowDistribution(noteData, num_arrows)
  arrows_long <- melt(arrows, id.var = 'arrow')
  arrow_labels <- c('Steps', 'Jumps', 'Brackets', 'Holds')
  max_freq <- max(rowSums(arrows))
  
  arrows_long$arrow <- as.factor(arrows_long$arrow)
  
  if(nchar(chartTitle) > 0) {
    chartTitle = paste0(' - ', chartTitle)
  }
  
  ggplot2::ggplot(arrows_long, aes(x = arrow, y = value, fill = variable)) +
    geom_bar(stat = 'identity', width = 0.7, color = 'lightblue', alpha = 1) +
    labs(y = 'Frequency', title = paste0('Step chart distribution', chartTitle)) +
    scale_y_continuous(expand = c(0,  0.05), limits = c(0, max_freq + 25)) +
    scale_fill_manual(breaks = waiver(), values = noteColors,
                      name = "Step type", labels = arrow_labels) +
    theme(axis.text.x = icon_axis(arrows = num_arrows, angle = 0),
          axis.title.x = element_blank(),
          legend.position = c(.1, .8),
          legend.background = element_rect(fill = '#131711', color = 'lightblue')) +      
    chartTheme
}

