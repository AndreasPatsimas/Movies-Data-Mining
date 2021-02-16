from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def bayes_model(countvect_reviews_train, countvect_reviews_test, train_sentiments, tfidf_vect_reviews_train, tfidf_vect_reviews_test):

    mnb = MultinomialNB()

    mnb_bow = mnb.fit(countvect_reviews_train, train_sentiments)
    print(mnb_bow)

    mnb_tfidf = mnb.fit(tfidf_vect_reviews_train, train_sentiments)
    print(mnb_tfidf)


    mnb_bow_predict = mnb.predict(countvect_reviews_test)
    print(mnb_bow_predict)

    mnb_tfidf_predict = mnb.predict(tfidf_vect_reviews_test)
    print(mnb_tfidf_predict)

    my_dictionary = {'mnb_bow_predict': mnb_bow_predict, 'mnb_tfidf_predict': mnb_tfidf_predict}

    return my_dictionary

def bayes_accuracy(test_sentiments, mnb_bow_predict, mnb_tfidf_predict):

    mnb_bow_score = accuracy_score(test_sentiments, mnb_bow_predict)
    print("mnb_bow_score :", mnb_bow_score)

    mnb_tfidf_score = accuracy_score(test_sentiments, mnb_tfidf_predict)
    print("mnb_tfidf_score :", mnb_tfidf_score)

def give_bayes_accuracy(test_sentiments, mnb_bow_predict, mnb_tfidf_predict):

    mnb_bow_score = accuracy_score(test_sentiments, mnb_bow_predict)

    mnb_tfidf_score = accuracy_score(test_sentiments, mnb_tfidf_predict)
    return mnb_bow_score, mnb_tfidf_score

def bayes_classification_report(test_sentiments, mnb_bow_predict, mnb_tfidf_predict):

    mnb_bow_report = classification_report(test_sentiments, mnb_bow_predict, target_names=['Positive', 'Negative'])
    print(mnb_bow_report)

    mnb_tfidf_report = classification_report(test_sentiments, mnb_tfidf_predict, target_names=['Positive', 'Negative'])
    print(mnb_tfidf_report)

def bayes_confusion_matrix(test_sentiments, mnb_bow_predict, mnb_tfidf_predict):

    cm_bow = confusion_matrix(test_sentiments, mnb_bow_predict, labels=[1, 0])
    print(cm_bow)

    cm_tfidf = confusion_matrix(test_sentiments, mnb_tfidf_predict, labels=[1, 0])
    print(cm_tfidf)

def bayes_execution(countvect_reviews_train, countvect_reviews_test, train_sentiments, tfidf_vect_reviews_train, tfidf_vect_reviews_test, test_sentiments):

    bayes_model_execution = bayes_model(countvect_reviews_train, countvect_reviews_test, train_sentiments, tfidf_vect_reviews_train,
                                  tfidf_vect_reviews_test)
    mnb_bow_predict = bayes_model_execution.get('mnb_bow_predict')
    mnb_tfidf_predict = bayes_model_execution.get('mnb_tfidf_predict')

    bayes_accuracy(test_sentiments, mnb_bow_predict, mnb_tfidf_predict)

    mnb_bow_score, mnb_tfidf_score = give_bayes_accuracy(test_sentiments, mnb_bow_predict, mnb_tfidf_predict)

    bayes_classification_report(test_sentiments, mnb_bow_predict, mnb_tfidf_predict)

    bayes_confusion_matrix(test_sentiments, mnb_bow_predict, mnb_tfidf_predict)

    return mnb_bow_score, mnb_tfidf_score