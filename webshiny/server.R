# shiny app logic

# load libraries
library(shiny)
library(ggplot2)
library(reticulate) # https://rstudio.github.io/reticulate/

CHART_LEVELS = list(pump_single = 26, pump_double = 28)

server <- function(input, output, session) {
  
  # update level slider
  observeEvent(input$chart_type, {
    n_levels = CHART_LEVELS[[input$chart_type]]
    updateSliderInput(session, 'chart_level', max = n_levels,
                      value = min(n_levels, input$chart_level))
  })
}

# to deploy to shinyapps.io
# setwd("C:/Users/Ryan/Documents/gmdev/projects/AICS/AICS/webshiny")
# library(rsconnect)
# deployApp(appdir) -> login to shinyapps