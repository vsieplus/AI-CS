## shiny

This directory contains code setting up the shiny app deployed [here](http://ec2-18-188-35-4.us-east-2.compute.amazonaws.com/aics/).
It provides functionality for chart generation, as well as tools for visualizing model outputs and the data itself.

You may also run it locally if you have R and shiny installed, with any models on your own machine. 
If you're currently in this directory, you can run the app from the command line by calling

```bash
R -e "shiny::runApp()"
```

If you do run it locally, you may change the value of `MODELS_DIR` in `generate.R` to wherever your
models are located on your system. The R packages used by the app are:

```bash
shiny
shinythemes
av
ggplot2
reticulate
```

In particular, the `reticulate` package provides an interface to python code. If you have
already setup an anaconda environment, you should also change the environment parameter
passed to `use_condaenv()`.
