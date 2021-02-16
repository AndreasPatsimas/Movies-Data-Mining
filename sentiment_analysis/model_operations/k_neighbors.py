from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def KNN_classifier(countvect_reviews_train, countvect_reviews_test, train_sentiments, tfidf_vect_reviews_train, tfidf_vect_reviews_test):

    knc = KNeighborsClassifier(n_neighbors=2)

    knc_bow = knc.fit(countvect_reviews_train, train_sentiments)
    print(knc_bow)

    knc_tfidf = knc.fit(tfidf_vect_reviews_train, train_sentiments)
    print(knc_tfidf)


    knc_bow_predict = knc.predict(countvect_reviews_test)
    print(knc_bow_predict)

    knc_tfidf_predict = knc.predict(tfidf_vect_reviews_test)
    print(knc_tfidf_predict)

    my_dictionary = {'knc_bow_predict': knc_bow_predict, 'knc_tfidf_predict': knc_tfidf_predict}

    return my_dictionary

def knc_accuracy(test_sentiments, knc_bow_predict, knc_tfidf_predict):

    knc_bow_score = accuracy_score(test_sentiments, knc_bow_predict)
    print("knc_bow_score :", knc_bow_score)

    knc_tfidf_score = accuracy_score(test_sentiments, knc_tfidf_predict)
    print("knc_tfidf_score :", knc_tfidf_score)
    
def knc_give_accuracy(test_sentiments, knc_bow_predict, knc_tf_idf_predict):

    knc_bow_score = accuracy_score(test_sentiments, knc_bow_predict)

    knc_tf_idf_score = accuracy_score(test_sentiments, knc_tf_idf_predict)
    return knc_bow_score, knc_tf_idf_score



def knc_classification_report(test_sentiments, knc_bow_predict, knc_tfidf_predict):

    knc_bow_report = classification_report(test_sentiments, knc_bow_predict, target_names=['Positive', 'Negative'])
    print(knc_bow_report)

    knc_tfidf_report = classification_report(test_sentiments, knc_tfidf_predict, target_names=['Positive', 'Negative'])
    print(knc_tfidf_report)


def knc_confusion_matrix(test_sentiments, knc_bow_predict, knc_tfidf_predict):

    knc_bow = confusion_matrix(test_sentiments, knc_bow_predict, labels=[1, 0])
    print(knc_bow)

    knc_tfidf = confusion_matrix(test_sentiments, knc_tfidf_predict, labels=[1, 0])
    print(knc_tfidf)

def knc_execution(countvect_reviews_train, countvect_reviews_test, train_sentiments, tfidf_vect_reviews_train, tfidf_vect_reviews_test, test_sentiments):

    randomf_forest_model_execution = KNN_classifier(countvect_reviews_train, countvect_reviews_test, train_sentiments, tfidf_vect_reviews_train,
                                  tfidf_vect_reviews_test)
    knc_bow_predict = randomf_forest_model_execution.get('knc_bow_predict')
    knc_tfidf_predict = randomf_forest_model_execution.get('knc_tfidf_predict')

    knc_accuracy(test_sentiments, knc_bow_predict, knc_tfidf_predict)
    
    knc_bow_score, knc_tf_idf_score = knc_give_accuracy(test_sentiments, knc_bow_predict,knc_tfidf_predict)

    knc_classification_report(test_sentiments, knc_bow_predict, knc_tfidf_predict)

    knc_confusion_matrix(test_sentiments, knc_bow_predict, knc_tfidf_predict)
    
    return knc_bow_score, knc_tf_idf_score