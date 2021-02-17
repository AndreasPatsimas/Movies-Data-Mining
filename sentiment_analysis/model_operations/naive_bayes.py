from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def bayes_model(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews):
    #training the model
    mnb = MultinomialNB()
    #fitting the svm for bag of words
    mnb_bow = mnb.fit(cv_train_reviews, train_sentiments)
    print(mnb_bow)
    #fitting the svm for tfidf features
    mnb_tfidf = mnb.fit(tv_train_reviews, train_sentiments)
    print(mnb_tfidf)

    #Predicting the model for bag of words
    mnb_bow_predict = mnb.predict(cv_test_reviews)
    print(mnb_bow_predict)
    #Predicting the model for tfidf features
    mnb_tfidf_predict = mnb.predict(tv_test_reviews)
    print(mnb_tfidf_predict)

    my_dictionary = {'mnb_bow_predict': mnb_bow_predict, 'mnb_tfidf_predict': mnb_tfidf_predict}

    return my_dictionary

def bayes_accuracy(test_sentiments, mnb_bow_predict, mnb_tfidf_predict):
    # Accuracy score for bag of words
    mnb_bow_score = accuracy_score(test_sentiments, mnb_bow_predict)
    print("mnb_bow_score :", mnb_bow_score)
    # Accuracy score for tfidf features
    mnb_tfidf_score = accuracy_score(test_sentiments, mnb_tfidf_predict)
    print("mnb_tfidf_score :", mnb_tfidf_score)

def give_bayes_accuracy(test_sentiments, mnb_bow_predict, mnb_tfidf_predict):
    # Accuracy score for bag of words
    mnb_bow_score = accuracy_score(test_sentiments, mnb_bow_predict)
    # Accuracy score for tfidf features
    mnb_tfidf_score = accuracy_score(test_sentiments, mnb_tfidf_predict)
    return mnb_bow_score, mnb_tfidf_score

def bayes_classification_report(test_sentiments, mnb_bow_predict, mnb_tfidf_predict):
    # Classification report for bag of words
    mnb_bow_report = classification_report(test_sentiments, mnb_bow_predict, target_names=['Positive', 'Negative'])
    print(mnb_bow_report)
    # Classification report for tfidf features
    mnb_tfidf_report = classification_report(test_sentiments, mnb_tfidf_predict, target_names=['Positive', 'Negative'])
    print(mnb_tfidf_report)

def bayes_confusion_matrix(test_sentiments, mnb_bow_predict, mnb_tfidf_predict):
    # confusion matrix for bag of words
    cm_bow = confusion_matrix(test_sentiments, mnb_bow_predict, labels=[1, 0])
    print(cm_bow)
    # confusion matrix for tfidf features
    cm_tfidf = confusion_matrix(test_sentiments, mnb_tfidf_predict, labels=[1, 0])
    print(cm_tfidf)

def bayes_execution(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments):

    bayes_model_execution = bayes_model(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews,
                                  tv_test_reviews)
    mnb_bow_predict = bayes_model_execution.get('mnb_bow_predict')
    mnb_tfidf_predict = bayes_model_execution.get('mnb_tfidf_predict')

    bayes_accuracy(test_sentiments, mnb_bow_predict, mnb_tfidf_predict)

    mnb_bow_score, mnb_tfidf_score = give_bayes_accuracy(test_sentiments, mnb_bow_predict, mnb_tfidf_predict)

    bayes_classification_report(test_sentiments, mnb_bow_predict, mnb_tfidf_predict)

    bayes_confusion_matrix(test_sentiments, mnb_bow_predict, mnb_tfidf_predict)

    return mnb_bow_score, mnb_tfidf_score