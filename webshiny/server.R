# shiny app logic

# load libraries/functions
library(shiny)

source('generate.R', local = TRUE)
source('visualize.R', local = TRUE)

server <- function(input, output, session) {
  
  
  # produce spectrogram plot for new audio file
  output$spectrogram_plot <- renderPlot({
    req(input$audio_file)
    plotSpectrogram(input$audio_file$datapath)
  }, height = 400, width = 600)
  
  # play audio if an audio file has been uploaded + user clicked play
  observeEvent(input$play_audio, {
    req(input$audio_file)
    output$audio <- renderUI(tags$audio(src = 'clip.wav', type = 'audio/wav',
                             autoplay = NA, controls = NA,
                             style = 'display: none;'))
  })
  
  # update model summary
  output$model_summary <- renderText({
    getModelSummary(input$model)
  })
  
  # update level slider and model options based on chart type
  observeEvent(input$chart_type, {
    n_levels <- CHART_LEVELS[[input$chart_type]]
    updateSliderInput(session, 'chart_level', max = n_levels)
    
    updateSelectInput(session, 'model', choices = modelsList[[input$chart_type]])
  })
}

# to deploy to shinyapps.io
# setwd("C:/Users/Ryan/Documents/gmdev/projects/AICS/AICS/webshiny")
# library(rsconnect)
# deployApp(appdir) -> login to shinyapps