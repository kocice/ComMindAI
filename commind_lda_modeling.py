#################################################################################################
# Import des librairies pour la modélisation de thématique
#################################################################################################
from pprint import pprint

# Gensim
# pip install --upgrade gensim --no-cache-dir
import gensim
import pandas as pd
from gensim.corpora import dictionary
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
# %matplotlib inline

from matplotlib import pyplot as plt

# Enable logging for gensim - optional
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TopicModeling:

    def __init__(self, list_sentences):
        # Create Dictionary
        self.sentence = list_sentences
        # Create a corpus from a list of texts
        self.id2word = dictionary.Dictionary(self.sentence)
        self.texts = self.sentence
        self.x = None

        # # Term Document Frequency
        self.corpus = [self.id2word.doc2bow(text) for text in self.sentence]

    def viewFreqWord(self):
        print([[(self.id2word[id], freq) for id, freq in cp] for cp in self.corpus[:1]])

    def compute_coherence_values(self, limit=40, start=2, step=6):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = gensim.models.ldamodel.LdaModel(corpus=self.corpus, num_topics=num_topics, id2word=self.id2word)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=self.texts, dictionary=self.id2word, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    def numTopicGraph(self, coherence_values):
        # Show graph
        limit = 40
        start = 2
        step = 6
        self.x = range(start, limit, step)
        # print(x, "\n")
        # print(coherence_values)
        plt.plot(self.x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend("coherence_values", loc='best')
        plt.show()
        return self.x

    def getBestNumTopic(self, coherence_values):
        # Print the coherence scores
        best_num_topic = 0
        # max_coherence = round(max(coherence_values), 4)
        coherence = 0
        for m, cv in zip(self.x, coherence_values):
            if round(cv, 4) < coherence:
                print(f"Num Topic: {m}  coherence score: {round(cv, 4)}")
                break
            best_num_topic += 1
            coherence = round(cv, 4)
        return best_num_topic

    def getOptimalModele(self, model_list, num_topic, words=10):
        # Select the model and print the topics
        optimal_model = model_list[num_topic]
        model_topics = optimal_model.show_topics(formatted=False)
        pprint(optimal_model.print_topics(num_words=words))
        print(optimal_model.num_topics)
        return optimal_model

    def makeLdaModel(self, num_topics):
        return gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                               id2word=self.id2word,
                                               num_topics=num_topics,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)

    def viewMetric(self, lda_model):
        # Compute Perplexity
        print('\nPerplexity: ',
              lda_model.log_perplexity(self.corpus))  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=self.sentence, dictionary=self.id2word,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)

    def showGraph(self, lda_model):
        # Visualize the topi
        pyLDAvis.enable_notebook()
        return pyLDAvis.gensim_models.prepare(lda_model, self.corpus, self.id2word)

    def getTopicData(self, ldamodel, df):
        # Init output
        sent_topics_df = pd.DataFrame()

        for i in range(len(self.corpus)):
            topic_stat = ldamodel.get_document_topics(self.corpus[i])
            topic_num, prop_topic = sorted(topic_stat, key=lambda x: x[1], reverse=True)[0]
            wp = ldamodel.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in wp])
            sent_topics_df = sent_topics_df.append(
                pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)

        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(df['comment'])
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1, ignore_index=True)
        sent_topics_df = pd.concat([sent_topics_df, df], axis=1, ignore_index=True)

        # Format
        sent_topics_df = sent_topics_df.reset_index()
        sent_topics_df.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords',
                                  'Text'] + df.columns.to_list()
        return sent_topics_df
