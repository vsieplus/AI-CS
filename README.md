# AI Custom Step

**AI Custom Step (AICS)** explores the use of neural networks to generate [Pump it Up](https://en.wikipedia.org/wiki/Pump_It_Up_(video_game_series)) 
custom step charts from audio. This project was inspired by [Dance Dance Convolution (DDC)](https://github.com/chrisdonahue/ddc).

## Repo overview
| Directory  | Description                                           |
|------------|-------------------------------------------------------|
| `data`     | Chart and song data retrieval + processing |
| `generate` | Chart generation process using a trained model |
| `shiny`    | Shiny app providing model and data interactivity + visualization |
| `train`    | Models and training scripts |

## Getting started

If you are mainly interested in generating a stepchart, you can try out some  of the pretrained models at 
the shiny web app [here](http://ec2-18-188-35-4.us-east-2.compute.amazonaws.com/aics/) (currently down b/c no funds for the server :-(). Simply upload an audio file, select the
type of chart you'd like, and it will generate a chart for you! There are also some useful visualization tools available.

If you are interested in trying to train your own models, or learning how the project works you can
continue reading below. To see the dependencies for the project refer to the `requirements.txt` file for python packages
and `shiny/` for R packages. Required versions:

- Python 3.8.x
- R 4.0.x

The pipeline for training a model and then using it to generate charts can be summarized as follows:

1. Use the scripts in `data/` to retrieve, extract, and process a particular collection of chart data for model training.

2. Use a specified dataset to train a pair of models in `train/`.

3. After training is complete, use the saved models to generate and save new charts in `generate/`

Each step consists of various sub-processes, so each directory itself has a README with further details.

### Shiny App Preview

![](https://raw.githubusercontent.com/vsieplus/AI-CS/master/res/prev1.PNG)
![](https://raw.githubusercontent.com/vsieplus/AI-CS/master/res/prev2.PNG)

### Issues/bugs

Feel free to raise any issues or bugs you may encounter, and ask any questions you might have. 
Thanks for reading!
