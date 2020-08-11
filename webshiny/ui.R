# ui.R
# shiny app simple ui

library(shiny)
library(shinythemes)

CHART_TYPES <- list(Single = 'pump-single', Double = 'pump-double')
CHART_FORMATS <- c('ssc', 'ucs')

ui <- navbarPage(
  windowTitle = 'Pump it Up - AI Custom Step',
  title = div(img(src = 'down_arrow.png', style = 'height: 27px; padding-right: 7px'), 
              'Pump it Up - AI Custom Step'), 
  position = 'static-top',
  theme = shinytheme('slate'),
  header = tags$head(includeCSS('style.css')),
  footer = tags$footer(includeHTML('html/footer.html')),
  
  # main tab
  tabPanel('Generate Steps',
    sidebarLayout(
      
      sidebarPanel(width = 3, position = 'left',
        selectInput('model', 'Select model', choices = c(1,2,3)),
        radioButtons('chart_type', 'Chart type:', choices = CHART_TYPES, inline = TRUE),
        sliderInput('chart_level', 'Chart level:', value = 1, min = 1, max = 26),
        fileInput('audio_file', 'Upload audio'),
        textInput('song_title', 'Song Title: '),
        textInput('artist', 'Artist: '),
        
        div(style = "vertical-align:top; display: inline-block", 
            checkboxGroupInput('save_formats', 'Output formats:',
                               choices = CHART_FORMATS, inline=TRUE)),
        div(style = "vertical-align:top; display: inline-block;
                     padding-left: 40px; padding-top: 16px", 
            actionButton('generate', ' Generate!', icon = icon('angle-up')))
      ),
      
      mainPanel(width = 9,
        includeHTML('html/generate_header.html'),
        textOutput('model_summary')
      )
    ),
  ),
  
  
  tabPanel('About',
    includeHTML('html/about.html')
  )
)