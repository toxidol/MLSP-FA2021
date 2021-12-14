# MLSP-FA2021
Course Project for 11755/18797

## Datasets
- TIMIT with clean speech
  - To download: `wget https://data.deepai.org/timit.zip`
  - Next: add noise to clean speech based on Bhiksha's paper
## Wiener
- Wiener_main adds white gaussian noise to clean speeches, enhance the noisy speeches, and calculate average SNR improvement.
- Wiener_as is the implementation of Wiener Filter
## References
- Wilson, Kevin W., Bhiksha Raj, Paris Smaragdis, and Ajay Divakaran. "Speech denoising using nonnegative matrix factorization with priors." In 2008 IEEE International Conference on Acoustics, Speech and Signal Processing, pp. 4029-4032. IEEE, 2008.
