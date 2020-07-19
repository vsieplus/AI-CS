# train

This directory contains code defining the models and training process.

### DDC

The neural models and training are largely based off of [DDC](https://github.com/chrisdonahue/ddc).
In particular, the task of 'learning to choreograph' is broken down into two
subtasks - step placement, and step selection. 

Step placement determines when
note(s) in a chart are to occur. ..


Step selection determines which note(s)
will occur. ..


### AICS

In this project, the step placement process remains largely unchanged. However, the step selection
process has some minor changes, simply as a result of the different dance pad layout.
Instead of 4 arrows to choose from at any given point, in PIU there are 5 and 10 for
singles and doubles charts, respectively. The consideration of doubles charts is
another difference from DDC. 

An additional step taken here is to also train charts to generate doubles charts.
This involves training a separate model in