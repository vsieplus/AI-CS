## shiny

This directory contains code setting up the shiny app deployed [here](https://vsie.shinyapps.io/ai_custom_step).
It provides functionality for chart generation, as well as tools for visualizing model outputs and the data itself.

You may also run it locally if you have R and shiny installed, with any models on your own machine. 
If you're currently in this directory, you can run the app from the command line by calling

```bash
R -e "shiny::runApp()"
```

If you do run it locally, be sure to change the value of `MODELS_DIR` in `generate.py` to wherever your
models are located on your system.
