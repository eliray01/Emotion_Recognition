# Emotion recognition
## Based on ECG, PPG and SGR

[![ECG](https://wallpaper.dog/large/5460308.png)]()

The interaction between the sympathetic and parasympathetic nervous systems significantly affects a person's emotional well-being. This research aims to understand the complex relationship between these two systems and emotional states. My part in this huge research is Emotion Recognition. Our study suggests an experimental approach in which negative emotions are artificially induced in the subjects, and their corresponding physiological reactions are monitored using electrocardiography (ECG), photoplethysmography (PPG) and galvanic skin response (GSR). In addition, we present a table of physiological indicators that establish links between the autonomic nervous system and emotional states.

Using machine learning methods, we have developed an innovative algorithm capable of recognizing emotional states based on physiological indicators. With this algorithm, we aim to provide a reliable and effective means of detecting emotions, paving the way for potential applications in various fields such as mental health, human-computer interaction and emotional computing.

Light GBM coped with this task best with F1 score 0.98. All the results obtained, as well as the dataset with novel feature extraction approach, that I created, can be found in this report.

## Models used

- Random Forest
- Gradient Boosting 
- LGM
- SVM

## Features

- Features extracted from ECG
- Features extracted from PPG
- Features extracted from SGR
- [TSFresh][TSFresh] features from ECG


***ALL ALGORIMTHS NECCESARY FOR EXTRACTING FEATURES FROM ECG, PPG and SGR ARE IN UTILS.PY***

```python 
from utils.utils import ppg_findpeaks_bishop, rr_interval,\
    exctract_hrv, exctract_hr, extract_p_wave,\
    extract_sgr, ppg_amplitude, ppg_frequency
```




## Supervisor

Research fellow, fut.PhD Alshanskaia Evgenia Igorevna


   [TSFresh]: <https://tsfresh.com/>
