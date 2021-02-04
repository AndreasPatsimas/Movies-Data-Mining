from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def random_forest_classifier(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews):
    # training the knc
    knc = KNeighborsClassifier(n_neighbors=2)
    # fitting the knc for bag of words
    knc_bow = knc.fit(cv_train_reviews, train_sentiments)
    print(knc_bow)
    # fitting the knc for tfidf features
    knc_tfidf = knc.fit(tv_train_reviews, train_sentiments)
    print(knc_tfidf)

    # Predicting the model for bag of words
    knc_bow_predict = knc.predict(cv_test_reviews)
    print(knc_bow_predict)
    # Predicting the model for tfidf features
    knc_tfidf_predict = knc.predict(tv_test_reviews)
    print(knc_tfidf_predict)

    my_dictionary = {'knc_bow_predict': knc_bow_predict, 'knc_tfidf_predict': knc_tfidf_predict}

    return my_dictionary

def knc_accuracy(test_sentiments, knc_bow_predict, knc_tfidf_predict):
    # Accuracy score for bag of words
    knc_bow_score = accuracy_score(test_sentiments, knc_bow_predict)
    print("knc_bow_score :", knc_bow_score)
    # Accuracy score for tfidf features
    knc_tfidf_score = accuracy_score(test_sentiments, knc_tfidf_predict)
    print("knc_tfidf_score :", knc_tfidf_score)
    
def knc_give_accuracy(test_sentiments, knc_bow_predict, knc_tf_idf_predict):
    # Accuracy score for bag of words
    knc_bow_score = accuracy_score(test_sentiments, knc_bow_predict)
    # Accuracy score for tfidf features
    knc_tf_idf_score = accuracy_score(test_sentiments, knc_tf_idf_predict)
    return knc_bow_score, knc_tf_idf_score



def knc_classification_report(test_sentiments, knc_bow_predict, knc_tfidf_predict):
    # Classification report for bag of words
    knc_bow_report = classification_report(test_sentiments, knc_bow_predict, target_names=['Positive', 'Negative'])
    print(knc_bow_report)
    # Classification report for tfidf features
    knc_tfidf_report = classification_report(test_sentiments, knc_tfidf_predict, target_names=['Positive', 'Negative'])
    print(knc_tfidf_report)


def knc_confusion_matrix(test_sentiments, knc_bow_predict, knc_tfidf_predict):
    # confusion matrix for bag of words
    knc_bow = confusion_matrix(test_sentiments, knc_bow_predict, labels=[1, 0])
    print(knc_bow)
    # confusion matrix for tfidf features
    knc_tfidf = confusion_matrix(test_sentiments, knc_tfidf_predict, labels=[1, 0])
    print(knc_tfidf)

def knc_execution(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments):

    randomf_forest_model_execution = random_forest_classifier(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews,
                                  tv_test_reviews)
    knc_bow_predict = randomf_forest_model_execution.get('knc_bow_predict')
    knc_tfidf_predict = randomf_forest_model_execution.get('knc_tfidf_predict')

    knc_accuracy(test_sentiments, knc_bow_predict, knc_tfidf_predict)
    
    knc_bow_score, knc_tf_idf_score = knc_give_accuracy(test_sentiments, knc_bow_predict,knc_tfidf_predict)

    knc_classification_report(test_sentiments, knc_bow_predict, knc_tfidf_predict)

    knc_confusion_matrix(test_sentiments, knc_bow_predict, knc_tfidf_predict)
    
    return knc_bow_score, knc_tf_idf_score