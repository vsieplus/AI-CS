# functions to visualize audio/chart properties

library(ggplot2)
library(av)

samplePath = file.path('www', 'clip.wav')

plotSpectrogram <- function(audio_path) {
  fft_data <- av::read_audio_fft(audio_path, overlap = 0.5,
                                 start_time = 45.0, end_time = 55.0)
  # save sample clip for output
  av::av_audio_convert(audio_path, samplePath, start_time = 45.0, total_time = 10.0)
  
  plot(fft_data)
}
