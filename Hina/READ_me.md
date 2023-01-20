# Multiclass Hate Speech Detection
## Data Understanding
Data preprocessing needs to be done by using steps of tokenisation, removing stopwords or specia character etc. before exploring the useful information contained in the data. The file count_words_per_class.ipynb summarizes the EDA part. Major analysis has been done regarding 
* distribution of classes
* distrubution of tokens per class 
* count of unique words per class
* representations of unique words via various shapes of wordcloud

## Modelling
BERT binary has been trained using 
* Pytorch Framework (BERT_token_200.ipynb)
* Tensor flow framework with HuggingFace Trainer class (sentiment_analysis_BERT_trainerclass.ipynb)

The notebooks contain concise comment son each step performed.

For deployment refer to [deployment_server](https://github.com/hinatanvir/etai_deployment_server)
