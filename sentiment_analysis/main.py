from sentiment_analysis.my_imports.import_data import import_data_csv
from sentiment_analysis.my_splits.split_data import split
from sentiment_analysis.methodologies.bag_of_words import execute_bow
from sentiment_analysis.methodologies.tf_idf import execute_tf_idf

from sentiment_analysis.model_operations.lr_regression import lr_regression_execution
from sentiment_analysis.model_operations.linear_svm import linear_svm_execution
from sentiment_analysis.model_operations.naive_bayes import bayes_execution
from sentiment_analysis.model_operations.k_neighbors import knc_execution
from sentiment_analysis.world_cloud_reviews.review import positive_review, negative_review

from sentiment_analysis.utils.text_format import denoise_text
from sentiment_analysis.utils.text_operations import remove_special_characters, simple_stemmer, remove_stopwords

from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
#for plots
from sentiment_analysis.plots.plots import plot_accuracies

my_data = import_data_csv()

split(my_data)

# pre-process
my_data['review'] = my_data['review'].apply(denoise_text)

my_data['review'] = my_data['review'].apply(remove_special_characters)

my_data['review'] = my_data['review'].apply(simple_stemmer)

#set stopwords/keywords to english
stop = set(stopwords.words('english'))
print(stop)

my_data['review']=my_data['review'].apply(remove_stopwords)

#normalized train reviews
norm_train_reviews = my_data.review[:40000]

#Normalized test reviews
norm_test_reviews = my_data.review[40000:]

# execute bag of words
bow_execution = execute_bow(norm_train_reviews, norm_test_reviews)
cv_train_reviews = bow_execution.get('cv_train_reviews')
cv_test_reviews = bow_execution.get('cv_test_reviews')

# execute tf-idf
tf_idf_execution = execute_tf_idf(norm_train_reviews, norm_test_reviews)
tv_train_reviews = tf_idf_execution.get('tv_train_reviews')
tv_test_reviews = tf_idf_execution.get('tv_test_reviews')

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
lr_regression_execution(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments)

# lr_regression execution
lr_bow_score, lr_tf_idf_score = lr_regression_execution(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments)

# linear svm
svm_bow_score, svm_tf_idf_score = linear_svm_execution(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments)

# Naive Bayes for bag of words and tfidf features
bayes_bow_score, bayes_tf_idf_score = bayes_execution(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments)

#KNeighbors
knc_bow_score, knc_tf_idf_score = knc_execution(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments)

# view with plot the positive review words
positive_review(norm_train_reviews)

# view with plot the negative review words
negative_review(norm_train_reviews)

#plot bow accuracies
accuracies = [lr_bow_score, svm_bow_score, bayes_bow_score,knc_bow_score]
algorithms = ['Rinear Regression', 'SVM', 'Bayes Classifier','KNeighbors Classifier']
plot_accuracies(accuracies,algorithms)

#plot tf_idf accuracies
accuracies = [ lr_tf_idf_score,  svm_tf_idf_score,  bayes_tf_idf_score, knc_tf_idf_score]
algorithms = ['Rinear Regression', 'SVM', 'Bayes Classifier', 'KNeighbors Classifier']
plot_accuracies(accuracies,algorithms)