from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def linear_svm_model(countvect_reviews_train, countvect_reviews_test, train_sentiments, tfidf_vect_reviews_train, tfidf_vect_reviews_test):

    svm = SGDClassifier(loss='hinge', max_iter=500, random_state=42)

    svm_bow = svm.fit(countvect_reviews_train, train_sentiments)
    print(svm_bow)

    svm_tfidf = svm.fit(tfidf_vect_reviews_train, train_sentiments)
    print(svm_tfidf)


    svm_bow_predict = svm.predict(countvect_reviews_test)
    print(svm_bow_predict)

    svm_tfidf_predict = svm.predict(tfidf_vect_reviews_test)
    print(svm_tfidf_predict)

    my_dictionary = {'svm_bow_predict': svm_bow_predict, 'svm_tfidf_predict': svm_tfidf_predict}

    return my_dictionary

def linear_svm_accuracy(test_sentiments, svm_bow_predict, svm_tfidf_predict):

    svm_bow_score = accuracy_score(test_sentiments, svm_bow_predict)
    print("svm_bow_score :", svm_bow_score)

    svm_tfidf_score = accuracy_score(test_sentiments, svm_tfidf_predict)
    print("svm_tfidf_score :", svm_tfidf_score)

def give_linear_svm_accuracy(test_sentiments, svm_bow_predict, svm_tfidf_predict):

    svm_bow_score = accuracy_score(test_sentiments, svm_bow_predict)

    svm_tfidf_score = accuracy_score(test_sentiments, svm_tfidf_predict)
    return svm_bow_score, svm_tfidf_score

def linear_svm_classification_report(test_sentiments, svm_bow_predict, svm_tfidf_predict):

    svm_bow_report = classification_report(test_sentiments, svm_bow_predict, target_names=['Positive', 'Negative'])
    print(svm_bow_report)

    svm_tfidf_report = classification_report(test_sentiments, svm_tfidf_predict, target_names=['Positive', 'Negative'])
    print(svm_tfidf_report)

def linear_svm_confusion_matrix(test_sentiments, svm_bow_predict, svm_tfidf_predict):

    cm_bow = confusion_matrix(test_sentiments, svm_bow_predict, labels=[1, 0])
    print(cm_bow)

    cm_tfidf = confusion_matrix(test_sentiments, svm_tfidf_predict, labels=[1, 0])
    print(cm_tfidf)

def linear_svm_execution(countvect_reviews_train, countvect_reviews_test, train_sentiments, tfidf_vect_reviews_train, tfidf_vect_reviews_test, test_sentiments):

    linear_svm_model_execution = linear_svm_model(countvect_reviews_train, countvect_reviews_test, train_sentiments, tfidf_vect_reviews_train,
                                  tfidf_vect_reviews_test)
    svm_bow_predict = linear_svm_model_execution.get('svm_bow_predict')
    svm_tfidf_predict = linear_svm_model_execution.get('svm_tfidf_predict')

    linear_svm_accuracy(test_sentiments, svm_bow_predict, svm_tfidf_predict)

    svm_bow_score, svm_tf_idf_score = give_linear_svm_accuracy(test_sentiments, svm_bow_predict, svm_tfidf_predict)

    linear_svm_classification_report(test_sentiments, svm_bow_predict, svm_tfidf_predict)

    linear_svm_confusion_matrix(test_sentiments, svm_bow_predict, svm_tfidf_predict)

    return svm_bow_score, svm_tf_idf_score