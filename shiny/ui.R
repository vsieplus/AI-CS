# ui.R
# shiny app simple ui

library(shiny)
library(shinythemes)

source('util.R', local = TRUE)

CHART_TYPES <- list(Single = 'single', Double = 'double')
CHART_FORMATS <- c('ssc', 'ucs')
SAMPLING_CHOICES <- c('top-k', 'top-p', 'greedy', 'multinom')

side_by_side <- function(x, second = FALSE, padding = '20px') {
  if(second) {
    div(style = paste0('vertical-align: top; display: 
                        inline-block; padding-left:', padding), x)
  } else {
    div(style = 'display: inline-block', x)
  }
}

ui <- navbarPage(
  windowTitle = 'Pump it Up - AI Custom Step',
  title = div(img(src = 'down_arrow.png', style = 'height: 27px; padding-right: 7px'), 
              'Pump it Up - AI Custom Step'), 
  position = 'static-top',
  theme = shinytheme('slate'),
  header = tags$head(includeCSS('www/style.css')),
  footer = tags$footer(includeHTML('html/footer.html')),
  
  # main tab
  tabPanel('Generate Steps',
    sidebarLayout(
     
      # Generation settings
      sidebarPanel(width = 3, position = 'left',
        radioButtons('chart_type', 'Chart/Model type:', choices = CHART_TYPES, inline = TRUE),
        selectInput('model', 'Select model', choices = modelsList[['single']]),
      
        sliderInput('chart_level', 'Chart level:', value = 1, min = 1, max = 26),
      
        fileInput('audio_file', 'Upload audio'),
        textInput('song_title', 'Song Title: '),
        textInput('artist', 'Artist: '),
      
        side_by_side(numericInput('bpm', 'BPM (optional):', min = 0, value = 120, width = '120px')),
        side_by_side(checkboxGroupInput('save_formats', 'Output formats:',
                                        choices = CHART_FORMATS, inline = TRUE), second = TRUE),

        side_by_side(radioButtons('sample_strat', 'Sampling strategy', choices = SAMPLING_CHOICES)),
        
        side_by_side(span(
          numericInput('topk_k', 'K value for top-k', value = 20, min = 0, max = 1024, width = '120px'),
          numericInput('topp_p', 'P value for top-p', min = 0, max = 1, value = 0.005, width = '120px')
        ), second = T),
      
        side_by_side(actionButton('generate_chart', 'Generate!', icon = icon('angle-right'))),
        side_by_side(downloadButton('download_chart', 'Download', icon = icon('arrow-down')),
                     second = TRUE, padding = '30px')
      ),
     
      # Display model info and visualizations
      mainPanel(width = 9,
        includeHTML('html/generate_header.html'),
        fluidRow(
          # column for model information
          column(4, h3('Current model'), htmlOutput('model_summary'), 
                    h3('Generation summary'), htmlOutput('gen_summary')),
          
          # column for audio, model, chart section visualizations
          column(5, 
           h3('Audio sample spectrogram'), h3(),
           actionButton('play_audio', 'Play audio clip', icon = icon('play')),
           plotOutput('spectrogram_plot'),
           # sliderInput (clip location) ?
           uiOutput('audio'),
           plotOutput('model_peak_picking_plot')
        ),

        fluidRow(
          # step chart/distribution visualization,
          column(7, plotOutput('chart_distribution_plot')),
          column(5, plotOutput('chart_section_plot')))
        )
        
      )
    )
  ),
  
  
  tabPanel('About',
           includeHTML('html/about.html')
  )
)