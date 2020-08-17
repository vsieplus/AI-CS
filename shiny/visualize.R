# functions to visualize audio/chart properties

library(dplyr)
library(ggplot2)
library(grid)
library(extrafont)
library(magick)
library(reshape2)
library(av)

library(reticulate) # change to path to your python version
parse <- import_from_path('parse', path = file.path('..', 'data', 'processing'))

# loadfonts(device = 'win') [run this the first time loading extrafont]
samplePath = file.path('www', 'clip.wav')

## AUDIO VISUALIZATION ##############################

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

########################################################

## CHART PROCESSING ####################################

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
    
    if(fileType == 'ssc') {
      noteData$step <- convertToUCS(noteData$step)
    }
    
    # values will be '-2' if no corresponding arrows found
    noteData$steps <- lapply(noteData$step, function(s) {
      unlist(gregexpr('[X]', s)) - 1
    })
    
    noteData$holds <- lapply(noteData$step, function(s) {
      unlist(gregexpr('[MHW]', s)) - 1
    })
    
    noteData
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

###########################################################

## CHART VISUALIZATION ####################################

notePositions <- list(top = '[1368]', center = '[27]', bottom = '[0459]')

# arrow icons to plot
notePaths <- rep(c('www/piudl.png', 'www/piuul.png', 'www/piuc.png',
                   'www/piuur.png', 'www/piudr.png'), 2)
names(notePaths) = seq(1, 10)
noteIcons <- sapply(notePaths, image_read)
noteColors <- c('#c3e9d0', '#8da6e2', '#6184d8', '#1e1014')

## START ####
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

# produce a plot of step chart section (i.e. a still-shot of a stepchart)
# noteData should be an m x 2 df, with observations of time/step
plotChartSection <- function(noteData, startTime = 45.0, endTime = 55.0, epsilon = 1e-5) {
  num_arrows <- nchar(as.character(noteData[1, 'step']))
  noteData <- filter(noteData, time >= startTime & time <= endTime)
  
  arrows <- seq(0, num_arrows - 1)
  arrow_positions <- lapply(arrows, function(a) {
    sapply(noteData$steps, function(s) a %in% s)
  })
  
  names(arrow_positions) <- arrows
  noteData <- cbind(noteData, arrow_positions)
  
  # list of arrow ids with a note for each timestep
  timeNotes <- lapply(noteData$time, function(time) {
    which(sapply(names(arrow_positions), function(a) {
      noteData[abs(noteData$time - time) < epsilon, a]
    })) - 1
  })
  
  noteTimes <- data.frame(time = sapply(noteData$time, function(t) {
    rep(t, length(timeNotes[[which(abs(noteData$time - t) < epsilon)]]))
  }), notes = unlist(timeNotes))
  
  # time vs notes
  ggplot2::ggplot(noteTimes, aes(x = notes, y = time)) +
    geom_point() + 
    theme(axis.text.x = icon_axis(arrows = num_arrows, angle = 0),
          axis.title.x = element_blank()) +
    scale_y_continuous(trans = 'reverse')
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

  arrows$steps <- filterNotes(arrows$arrow, function(a) { 
    sum(sapply(noteData$steps, function(p) {
      length(p) == 1 && a == p[[1]]
    }))
  })
  
  arrows$jumps <- filterNotes(arrows$arrow, function(a) {
    sum(sapply(noteData$steps, function(s) length(s) == 2 & a %in% s) & 
      sapply(noteData$holds, function(h) all(h == -2)))
  })
  
  arrows$brackets <- filterNotes(arrows$arrow, function(a) {
    sum(sapply(noteData$steps, function(s) length(s) >= 3 & a %in% s) | 
      (sapply(noteData$holds, function(h) all(h != -2)) & 
       sapply(noteData$steps, function(s) {
        length(s) == 2 & a %in% s 
       })))
  })
  
  arrows$holds <- sapply(arrows$arrow, function(a) { 
    sum(sapply(noteData$holds, function(p) a %in% p))
  })
  
  arrows
}

# Plot distribution of chart steps
plotChartDistribution <- function(noteData, chartTitle = '') {
  # filter empty steps + extract note positions
  num_arrows <- nchar(as.character(noteData[1, 'step']))
  noteData <- dplyr::filter(noteData, step != strrep('.', num_arrows))
  
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
    labs(y = 'Frequency', title = paste0('Step Chart Distribution', chartTitle)) +
    scale_y_continuous(expand = c(0,  0.05), limits = c(0, max_freq + 25)) +
    scale_fill_manual(breaks = waiver(), values = noteColors,
                      name = "Step type", labels = arrow_labels) +
    theme(text = element_text(family = 'Calibri', color = 'white', size = 12),
          plot.title = element_text(size = 16),
          plot.background = element_rect(fill = '#131711', color = 'lightblue'),
          panel.background = element_rect(fill = 'black', color = 'lightblue'),
          panel.grid.major.y = element_line(color='gray50'),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.y = element_line(color = 'gray30'),
          panel.grid.minor = element_blank(),
          axis.text.x = icon_axis(arrows = num_arrows, angle = 0),
          axis.title.x = element_blank(),
          axis.text.y = element_text(color = 'white'),
          legend.position = c(.1, .8),
          legend.background = element_rect(fill = '#131711', color = 'lightblue'))
}

