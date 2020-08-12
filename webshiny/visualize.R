# functions to visualize audio/chart properties

library(ggplot2)
library(tuneR)
library(signal)

plot_spectrogram <- function(audio_path) {
  file_extension <- strsplit(audio_path, '.', fixed=TRUE)[[1]][-1]
  if(file_extension == 'mp3') {
    audio_data <- tuneR::readMP3(audio_path)
  } else if(file_extension == 'wav') {
    audio_data <- tuneR::readWave(audio_path)    
  } else {
    image()
  }
  
  # remove dc bias
  dc_balanced <- audio_data@left - mean(audio_data@left)

  # 3 features; S ~ FFT output, f ~ frequency (row), t ~ time (col)  
  spec <- signal::specgram(x = dc_balanced, n = 1024,
                           Fs = audio_data@samp.rate)
  
  # normalize, choose dynamic range
  S <- abs(spec$S)
  S <- S / max(S)
  S <- max(S, 10^(-60/10))
  S <- min(S, 10^(-3/10))
  
  out <- pmax(1e-6, S) # add 1e-6 to '0' vals
  dim(out) <- dim(S)
  out <- log10(out) / log10(1e-6)
  
  image(x = spec$t, y = spec$f, z = t(out), useRaster = TRUE,
        col = hcl.colors(n = 12, palette = 'Blues 2'),
        xlab = 'Time (s)', ylab = 'Freq (Hz)')
}
