# shiny app logic

# load libraries/functions
library(shiny)

source('generate.R', local = TRUE)
source('visualize.R', local = TRUE)

server <- function(input, output, session) {
  chartData <- NULL
  
  # update model summary text
  output$model_summary <- renderText({
    getModelSummary(input$model, as_str = TRUE)
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

    # track generation progress
    progress <- shiny::Progress$new()
    progress$set(message = 'Generating chart..', value = 0)
    on.exit(progress$close())

    updateProgress <- function(value , detail) {
      progress$set(value = value, detail = detail)
    }

    generateChart(input$audio_file$datapath, input$model, input$chart_level,
                  input$chart_type, input$song_title, input$artist, input$bpm,
                  input$save_formats, updateProgress)
  })
  
  # progress bar(s) (generation complete, saving complete)
  
  # save the chart once it's been generated
  output$download_chart <- downloadHandler(
    filename = function() {
      if(is.null(chartData)) {
        ''
      } else {
        paste0(chartData()[['name']], '.zip') 
      }
    },
    content = function(file) {
      if(is.null(chartData)) {
        return(NULL)
      }

      # can access reactive 'chartData' list in here      
      # use temp directory before zipping file
      chart_df = chartData()
      origDir = setwd(tempdir())
      on.exit(setwd(origDir))
      
      if(length(chart_df[['saveFormats']]) > 1) {
        saveFormat = 'both'
      } else {
        saveFormat = chart_df[['saveFormats']][1]
      }
      
      generate$save_chart(chart_df[['notes']], chart_df[['chartType']], chart_df[['level']],
                          saveFormat, chart_df[['title']], chart_df[['artist']],
                          chart_df[['audioPath']], chart_df[['name']], '.')
      
      zip(file, list.files('.', pattern = '(\\.ucs|\\.mp3|\\.ssc)'))  
    }
  )
  
  # produce model peak-picking plot
  output$model_peak_picking_plot <- renderPlot({
    plotPeakPicking(chartData()[['peaks']], chartData()[['thresholds']])
  }, height = 360, width = 600)
  
  # produce chart visualizations
  output$chart_section_plot <- renderPlot({
    plotChartSection(chartData()[['notes']])
  }, height = 300, width = 600)
  
  output$chart_distribution_plot <- renderPlot({
    plotChartDistribution(chartData()[['notes']])
  }, height = 500, width = 900)
}

# to deploy to shinyapps.io
# setwd("where this file is located"")
# library(rsconnect)
# deployApp(appdir) -> login to shinyapps