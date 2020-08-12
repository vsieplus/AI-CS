# ui.R
# shiny app simple ui

library(shiny)
library(shinythemes)

CHART_TYPES <- list(Single='pump-single', Double='pump-double')
CHART_FORMATS <- c('ssc', 'u <- cs')

# you can change this (relative path) to another folder containing
# your models when running locally
MODELS_DIR = 'models/'
MODELS_LIST = list.dirs(MODELS_DIR, full.names=FALSE)

side_by_side <- function(x, second=FALSE) {
  if(second) {
    div(style='vertical-align:top; padding-left: 20px; display: inline-block', x)
  } else {
    div(style='vertical-align:center; display: inline-block', x)
  }
}

ui <- navbarPage(
  windowTitle='Pump it Up - AI Custom Step',
  title=div(img(src='down_arrow.png', style='height: 27px; padding-right: 7px'), 
            'Pump it Up - AI Custom Step'), 
  position='static-top',
  theme=shinytheme('slate'),
  header=tags$head(includeCSS('www/style.css')),
  footer=tags$footer(includeHTML('html/footer.html')),
  
  # main tab
  tabPanel('Generate Steps',
    sidebarLayout(
      
      # Generation settings
      sidebarPanel(width=3, position='left',
        selectInput('model', 'Select model', choices=MODELS_LIST),
        
        radioButtons('chart_type', 'Chart type:', choices=CHART_TYPES, inline=TRUE),
        sliderInput('chart_level', 'Chart level:', value=1, min=1, max=26),
        
        fileInput('audio_file', 'Upload audio'),
        textInput('song_title', 'Song Title: '),
        textInput('artist', 'Artist: '),
        
        side_by_side(numericInput('bpm', 'BPM (optional):', min=0, value=120, width='120px')),
        side_by_side(checkboxGroupInput('save_formats', 'Output formats:',
                                        choices=CHART_FORMATS, inline=TRUE), second=TRUE),
        
        actionButton('generate', 'Generate!', icon=icon('angle-up'))
      ),
      
      # Display model info and visualizations
      mainPanel(width=9,
        includeHTML('html/generate_header.html'),
        textOutput('model_summary'),
        
        
        plotOutput('spectrogram_visual'),
        plotOutput('chart_visual')
      )
    ),
  ),
  
  
  tabPanel('About',
    includeHTML('html/about.html')
  )
)