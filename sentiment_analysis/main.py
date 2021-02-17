from sentiment_analysis.my_imports.import_data import import_data_csv
from sentiment_analysis.my_splits.split_data import split
from sentiment_analysis.methodologies.bag_of_words import execute_bag_of_words
from sentiment_analysis.methodologies.tf_idf import execute_tf_idf

from sentiment_analysis.model_operations.lr_regression import lr_regression_execution
from sentiment_analysis.model_operations.linear_svm import linear_svm_execution
from sentiment_analysis.model_operations.naive_bayes import bayes_execution
from sentiment_analysis.model_operations.k_neighbors import knc_execution
from sentiment_analysis.world_cloud_reviews.review import positive_review, negative_review

from sentiment_analysis.utils.text_format import remove_text_noise
from sentiment_analysis.utils.text_operations import drop_special_characters, text_streamer, drop_stopwords
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
#for plots
from sentiment_analysis.plots.plots import plot_accuracies
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

my_data = import_data_csv()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,8))
text_len=my_data[my_data['sentiment'] == 'positive']['review'].str.split().map(lambda x: len(x))
ax1.hist(text_len,color='purple')
ax1.set_title('Positive Reviews')
text_len=my_data[my_data['sentiment'] == 'negative']['review'].str.split().map(lambda x: len(x))
ax2.hist(text_len,color='blue')
ax2.set_title('Negative Reviews')
fig.suptitle('Words per sentence')
plt.show()

sns.set(style = "darkgrid" , font_scale = 1.2)
sns.countplot(my_data.sentiment)

split(my_data)

# pre-process
my_data['review'] = my_data['review'].apply(remove_text_noise)

my_data['review'] = my_data['review'].apply(drop_special_characters)

my_data['review'] = my_data['review'].apply(text_streamer)

#set stopwords to english
stop = set(stopwords.words('english'))
print(stop)

my_data['review']=my_data['review'].apply(drop_stopwords)

normal_reviews_train = my_data.review[:40000]

normal_reviews_test = my_data.review[40000:]

# execute bag of words
bag_of_words_execution = execute_bag_of_words(normal_reviews_train, normal_reviews_test)
count_vect_train_reviews = bag_of_words_execution.get('cv_train_reviews')
count_vect_test_reviews = bag_of_words_execution.get('cv_test_reviews')

# execute tf-idf
tf_idf_execution = execute_tf_idf(normal_reviews_train, normal_reviews_test)
tf_idf_vect_train_reviews = tf_idf_execution.get('tv_train_reviews')
tf_idf_vect_test_reviews = tf_idf_execution.get('tv_test_reviews')

#labeling the sentiment data
lb=LabelBinarizer()
#transformed sentiment data
sentiment_data = lb.fit_transform(my_data['sentiment'])
print(sentiment_data.shape)

#Spliting the sentiment data
train_sentiments = sentiment_data[:40000]
test_sentiments = sentiment_data[40000:]
print(train_sentiments)
print(test_sentiments)

# lr_regression execution
lr_bow_score, lr_tf_idf_score = lr_regression_execution(count_vect_train_reviews, count_vect_test_reviews, train_sentiments, tf_idf_vect_train_reviews, tf_idf_vect_test_reviews, test_sentiments)

# linear svm
svm_bow_score, svm_tf_idf_score = linear_svm_execution(count_vect_train_reviews, count_vect_test_reviews, train_sentiments, tf_idf_vect_train_reviews, tf_idf_vect_test_reviews, test_sentiments)

# Naive Bayes for bag of words and tfidf features
bayes_bow_score, bayes_tf_idf_score = bayes_execution(count_vect_train_reviews, count_vect_test_reviews, train_sentiments, tf_idf_vect_train_reviews, tf_idf_vect_test_reviews, test_sentiments)

#KNeighbors
knc_bow_score, knc_tf_idf_score = knc_execution(count_vect_train_reviews, count_vect_test_reviews, train_sentiments, tf_idf_vect_train_reviews, tf_idf_vect_test_reviews, test_sentiments)

############# tensorflow ##################################################
y = sentiment_data
X = my_data.drop(labels=['sentiment'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2001)

s = my_data['review'].to_numpy()
print(s)

embed = hub.load("https://tfhub.dev/google/nnlm-en-dim50/2")
embeddings = embed(s)

hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2", input_shape=[], dtype=tf.string)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    X_train,
    y_train,
    epochs=8,
    validation_data=(X_test, y_test)
)

results = model.evaluate(X_test, y_test)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

#############################################################################

# view with plot the positive review words
positive_review(normal_reviews_train)

# view with plot the negative review words
negative_review(normal_reviews_train)

#plot bow accuracies
accuracies = [lr_bow_score, svm_bow_score, bayes_bow_score,knc_bow_score]
algorithms = ['Logistic Regression', 'SVM', 'Bayes Classifier','KNeighbors Classifier']
plot_accuracies(accuracies,algorithms)

#plot tf_idf accuracies
accuracies = [ lr_tf_idf_score,  svm_tf_idf_score,  bayes_tf_idf_score, knc_tf_idf_score]
algorithms = ['Logistic Regression', 'SVM', 'Bayes Classifier', 'KNeighbors Classifier']
plot_accuracies(accuracies,algorithms)