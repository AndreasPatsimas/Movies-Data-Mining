from sentiment_analysis.my_imports.import_data import import_data_csv
from sentiment_analysis.my_splits.split_data import split
from sentiment_analysis.methodologies.bag_of_words import execute_bow
from sentiment_analysis.methodologies.tf_idf import execute_tf_idf

from sentiment_analysis.model_operations.lr_regression import lr_regression_execution
from sentiment_analysis.model_operations.linear_svm import linear_svm_execution
from sentiment_analysis.model_operations.naive_bayes import bayes_execution
from sentiment_analysis.world_cloud_reviews.review import positive_review, negative_review

from sentiment_analysis.utils.text_format import denoise_text
from sentiment_analysis.utils.text_operations import remove_special_characters, simple_stemmer, remove_stopwords

from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords

my_data = import_data_csv()

split(my_data)

my_data['review'] = my_data['review'].apply(denoise_text)

my_data['review'] = my_data['review'].apply(remove_special_characters)

my_data['review'] = my_data['review'].apply(simple_stemmer)

#set stopwords to english
stop = set(stopwords.words('english'))
print(stop)

my_data['review']=my_data['review'].apply(remove_stopwords)

#normalized train reviews
norm_train_reviews = my_data.review[:40000]
print(norm_train_reviews[0])

#Normalized test reviews
norm_test_reviews = my_data.review[40000:]
print(norm_test_reviews[45005])

# execute bag of words
bow_execution = execute_bow(norm_train_reviews, norm_test_reviews)
cv_train_reviews = bow_execution.get('cv_train_reviews')
cv_test_reviews = bow_execution.get('cv_test_reviews')

# execute tf-idf
tf_idf_execution = execute_tf_idf(norm_train_reviews, norm_test_reviews)
tv_train_reviews = tf_idf_execution.get('tv_train_reviews')
tv_test_reviews = tf_idf_execution.get('tv_test_reviews')

#labeling the sentient data
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

# linear svm
linear_svm_execution(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments)

# Naive Bayes for bag of words and tfidf features
bayes_execution(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments)

# word cloud for positive review words
positive_review(norm_train_reviews)

# word cloud for negative review words
negative_review(norm_train_reviews)