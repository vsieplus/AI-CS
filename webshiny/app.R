# code setting up the shiny app ui/logic

# load libraries
library(shiny)
library(ggplot2)
library(reticulate)

ui <- fluidPage(
    theme = 'style.css',

    h1(tags$strong('Pump it Up - AI Custom Step')),

    tabsetPanel(
        tabPanel('Generate Steps'),
        tabPanel('About'),
    )
)

server <- function(input, output, session) {

}

shinyApp(ui = ui, server = server)

# to deploy
# library(rsconnect)
# deployApp(appdir) -> login to shinyapps