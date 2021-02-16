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
from sentiment_analysis.plots.plots import plot_accuracies
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

my_data = import_data_csv()

sns.set(style = "darkgrid" , font_scale = 1.2)
sns.countplot(my_data.sentiment)

split(my_data)

# Προ-επεξεργασία Δεδομένων
my_data['review'] = my_data['review'].apply(remove_text_noise)

my_data['review'] = my_data['review'].apply(drop_special_characters)

my_data['review'] = my_data['review'].apply(text_streamer)

#Stop Words στα αγγλικά =
stop = set(stopwords.words('english'))
print(stop)

my_data['review']=my_data['review'].apply(drop_stopwords)

normal_train_reviews = my_data.review[:40000]

normal_test_reviews = my_data.review[40000:]

# bag of words
bag_of_words_execution = execute_bag_of_words(normal_train_reviews, normal_test_reviews)
countvect_reviews_train = bag_of_words_execution.get('countvect_reviews_train')
countvect_reviews_test = bag_of_words_execution.get('countvect_reviews_test')

# tf-idf
tf_idf_execution = execute_tf_idf(normal_train_reviews, normal_test_reviews)
tv_train_reviews = tf_idf_execution.get('tv_train_reviews')
tv_test_reviews = tf_idf_execution.get('tv_test_reviews')

#labeling the sentiment data
lb=LabelBinarizer()
#transformed sentiment data
sentiment_data = lb.fit_transform(my_data['sentiment'])
print(sentiment_data.shape)

#Διαχωρισμός σε test και train με αναλογία 80% - 20%
train_sentiments = sentiment_data[:40000]
test_sentiments = sentiment_data[40000:]
print(train_sentiments)
print(test_sentiments)

# logistic_regression
lr_bow_score, lr_tf_idf_score = lr_regression_execution(countvect_reviews_train, countvect_reviews_test, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments)

# linear support vector machine
svm_bow_score, svm_tf_idf_score = linear_svm_execution(countvect_reviews_train, countvect_reviews_test, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments)

# Naive Bayes
bayes_bow_score, bayes_tf_idf_score = bayes_execution(countvect_reviews_train, countvect_reviews_test, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments)

#KNN
knc_bow_score, knc_tf_idf_score = knc_execution(countvect_reviews_train, countvect_reviews_test, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments)

############# Artificial Neural Network ##################################################
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

# WordCloud για θετικά reviews
positive_review(normal_train_reviews)

# WordCloud για αρνητικά reviews
negative_review(normal_train_reviews)

#Διαγραμματική αναπαράσταση των accuracies για τον αλγόριθμο bag of words
accuracies = [lr_bow_score, svm_bow_score, bayes_bow_score,knc_bow_score]
algorithms = ['Logistic Regression', 'SVM', 'Bayes Classifier','KNeighbors Classifier']
plot_accuracies(accuracies,algorithms)

#Διαγραμματική αναπαράσταση των accuracies για τον αλγόριθμο tfidf
accuracies = [ lr_tf_idf_score,  svm_tf_idf_score,  bayes_tf_idf_score, knc_tf_idf_score]
algorithms = ['Logistic Regression', 'SVM', 'Bayes Classifier', 'KNeighbors Classifier']
plot_accuracies(accuracies,algorithms)