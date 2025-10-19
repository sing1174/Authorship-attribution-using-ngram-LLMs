# Authorship-attribution-using-ngram-LMs

### Generative and Discriminative Authorship Classifier
Project Overview
This project implements both generative and discriminative models to classify the author of given text samples. The generative model relies on n-gram language modeling techniques, while the discriminative model uses HuggingFace's pre-trained DistilBERT for text classification. The goal is to accurately predict whether a text belongs to one of the following authors: Jane Austen, Charles Dickens, Leo Tolstoy, or Oscar Wilde.

### Project Structure
classifier.py: Contains the code for the generative authorship classifier as well as discriminative classifier. 

##### Train and evaluate Generative Classifier (built using using NLTK-based n-gram models)
1. **Train on 90% of the dataset and evaluate on the reamining 10% of dataset**

   ```bash
   $ python classifier.py authorlist -approach generative
3. **Train on full dataset and evaluate on the test_sents.txt file**
   ```bash
   $ python classifier.py authorlist -approach generative -test test_sents.txt
   
##### For DistilBERT based classifier
1. **Train on 90% of the dataset and evaluate on the reamining 10% of dataset**

   ```bash
   $ python classifier.py authorlist -approach discriminative
2. **Train on full dataset and evaluate on the test_sents.txt file**
   ```bash
   $ python classifier.py authorlist -approach discriminative -test test_sents.txt

#### Text generation using NLTK models
Checkout [text generation](https://github.com/1998anwesha/Authorship-attribution-using-ngram-LMs/blob/main/text_generation_analysis.ipynb) results from n-gram models of each author. For five given prompts, one sample per author with corresponding perplexity score is reported. 
