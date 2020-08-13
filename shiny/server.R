# shiny app logic

# load libraries/functions
library(shiny)

source('generate.R', local = TRUE)
source('visualize.R', local = TRUE)

server <- function(input, output, session) {
  
  # update model summary text
  output$model_summary <- renderText({
    getModelSummary(input$model)
  })
  
  # update level slider and model options based on chart type
  observeEvent(input$chart_type, {
    n_levels <- CHART_LEVELS[[input$chart_type]]
    updateSliderInput(session, 'chart_level', max = n_levels)
    
    updateSelectInput(session, 'model', choices = modelsList[[input$chart_type]])
  })
  
  # produce spectrogram plot for new audio file
  output$spectrogram_plot <- renderPlot({
    req(input$audio_file)
    plotSpectrogram(input$audio_file$datapath)
  }, height = 360, width = 600)
  
  # play audio if an audio file has been uploaded + user clicked play
  observeEvent(input$play_audio, {
    req(input$audio_file)
    output$audio <- renderUI(tags$audio(src = 'clip.wav', type = 'audio/wav',
                             autoplay = NA, controls = NA,
                             style = 'display: none;'))
  })
  
  # perform chart generation once the audio file is uploaded
  # store model outputs + the generated notes/times
  chartData <- eventReactive(input$generate_chart, {
    req(input$audio_file)
    generateChart(input$audio_file$datapath, input$model, input$chart_level,
                  input$chart_type, input$song_title, input$artist, input$bpm,
                  input$save_formats)
  })
  
  # progress bar(s) (generation complete, saving complete)
  
  # save the chart once it's been generated
  output$download_chart <- downloadHandler(
    filename = function () {
      if(length(chartData[['saveFormats']]) > 1) {
        ext = '.zip'
      } else {
        ext = chartData[['saveFormats']][0]
      }
      
      # use same name as audio file
      paste0(chartData[['name']], ext)
    },
    content = saveCharts
  )
  
  # produce model output plots after generation
  
  
  # produce chart plots
  output$chart_section_plot <- renderPlot({
    plotChartSection(chartData[['notes']])
  }, height = 360, width = 600)
  
  output$chart_distribution_plot <- renderPlot({
    plotChartDist(chartData[['notes']])
  }, height = 400, width = 600)
}

# to deploy to shinyapps.io
# setwd("C:/Users/Ryan/Documents/gmdev/projects/AICS/AICS/shiny")
# library(rsconnect)
# deployApp(appdir) -> login to shinyapps