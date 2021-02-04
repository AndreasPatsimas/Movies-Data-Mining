from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def linear_svm_model(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews):
    #training the linear svm
    svm = SGDClassifier(loss='hinge', max_iter=500, random_state=42)
    #fitting the svm for bag of words
    svm_bow = svm.fit(cv_train_reviews, train_sentiments)
    print(svm_bow)
    #fitting the svm for tfidf features
    svm_tfidf = svm.fit(tv_train_reviews, train_sentiments)
    print(svm_tfidf)

    #Predicting the model for bag of words
    svm_bow_predict = svm.predict(cv_test_reviews)
    print(svm_bow_predict)
    #Predicting the model for tfidf features
    svm_tfidf_predict = svm.predict(tv_test_reviews)
    print(svm_tfidf_predict)

    my_dictionary = {'svm_bow_predict': svm_bow_predict, 'svm_tfidf_predict': svm_tfidf_predict}

    return my_dictionary

def linear_svm_accuracy(test_sentiments, svm_bow_predict, svm_tfidf_predict):
    # Accuracy score for bag of words
    svm_bow_score = accuracy_score(test_sentiments, svm_bow_predict)
    print("svm_bow_score :", svm_bow_score)
    # Accuracy score for tfidf features
    svm_tfidf_score = accuracy_score(test_sentiments, svm_tfidf_predict)
    print("svm_tfidf_score :", svm_tfidf_score)

def give_linear_svm_accuracy(test_sentiments, svm_bow_predict, svm_tfidf_predict):
    # Accuracy score for bag of words
    svm_bow_score = accuracy_score(test_sentiments, svm_bow_predict)
    # Accuracy score for tfidf features
    svm_tfidf_score = accuracy_score(test_sentiments, svm_tfidf_predict)
    return svm_bow_score, svm_tfidf_score

def linear_svm_classification_report(test_sentiments, svm_bow_predict, svm_tfidf_predict):
    # Classification report for bag of words
    svm_bow_report = classification_report(test_sentiments, svm_bow_predict, target_names=['Positive', 'Negative'])
    print(svm_bow_report)
    # Classification report for tfidf features
    svm_tfidf_report = classification_report(test_sentiments, svm_tfidf_predict, target_names=['Positive', 'Negative'])
    print(svm_tfidf_report)

def linear_svm_confusion_matrix(test_sentiments, svm_bow_predict, svm_tfidf_predict):
    # confusion matrix for bag of words
    cm_bow = confusion_matrix(test_sentiments, svm_bow_predict, labels=[1, 0])
    print(cm_bow)
    # confusion matrix for tfidf features
    cm_tfidf = confusion_matrix(test_sentiments, svm_tfidf_predict, labels=[1, 0])
    print(cm_tfidf)

def linear_svm_execution(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments):

    linear_svm_model_execution = linear_svm_model(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews,
                                  tv_test_reviews)
    svm_bow_predict = linear_svm_model_execution.get('svm_bow_predict')
    svm_tfidf_predict = linear_svm_model_execution.get('svm_tfidf_predict')

    linear_svm_accuracy(test_sentiments, svm_bow_predict, svm_tfidf_predict)

    svm_bow_score, svm_tf_idf_score = give_linear_svm_accuracy(test_sentiments, svm_bow_predict, svm_tfidf_predict)

    linear_svm_classification_report(test_sentiments, svm_bow_predict, svm_tfidf_predict)

    linear_svm_confusion_matrix(test_sentiments, svm_bow_predict, svm_tfidf_predict)

    return svm_bow_score, svm_tf_idf_score