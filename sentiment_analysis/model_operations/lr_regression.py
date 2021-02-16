from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def lr_model(countvect_reviews_train, countvect_reviews_test, train_sentiments, tfidf_vect_reviews_train, tfidf_vect_reviews_test):

    lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)


    lr_bow = lr.fit(countvect_reviews_train, train_sentiments)
    print(lr_bow)


    lr_tf_idf = lr.fit(tfidf_vect_reviews_train, train_sentiments)
    print(lr_tf_idf)


    lr_bow_predict = lr.predict(countvect_reviews_test)
    print(lr_bow_predict)


    lr_tf_idf_predict = lr.predict(tfidf_vect_reviews_test)
    print(lr_tf_idf_predict)

    my_dictionary = {'lr_bow_predict': lr_bow_predict, 'lr_tf_idf_predict': lr_tf_idf_predict}

    return my_dictionary

def lr_accuracy(test_sentiments, lr_bow_predict, lr_tf_idf_predict):

    lr_bow_score = accuracy_score(test_sentiments, lr_bow_predict)
    print("lr_bow_score :", lr_bow_score)

    lr_tf_idf_score = accuracy_score(test_sentiments, lr_tf_idf_predict)
    print("lr_tfidf_score :", lr_tf_idf_score)

def lr_give_accuracy(test_sentiments, lr_bow_predict, lr_tf_idf_predict):

    lr_bow_score = accuracy_score(test_sentiments, lr_bow_predict)

    lr_tf_idf_score = accuracy_score(test_sentiments, lr_tf_idf_predict)
    return lr_bow_score, lr_tf_idf_score

def lr_classification_report(test_sentiments, lr_bow_predict, lr_tf_idf_predict):
    lr_bow_report = classification_report(test_sentiments, lr_bow_predict, target_names=['Positive', 'Negative'])
    print(lr_bow_report)

    lr_tfidf_report = classification_report(test_sentiments, lr_tf_idf_predict, target_names=['Positive', 'Negative'])
    print(lr_tfidf_report)

def lr_confusion_matrix(test_sentiments, lr_bow_predict, lr_tf_idf_predict):

    cm_bow = confusion_matrix(test_sentiments, lr_bow_predict, labels=[1, 0])
    print(cm_bow)

    cm_tfidf = confusion_matrix(test_sentiments, lr_tf_idf_predict, labels=[1, 0])
    print(cm_tfidf)

def lr_regression_execution(countvect_reviews_train, countvect_reviews_test, train_sentiments, tfidf_vect_reviews_train, tfidf_vect_reviews_test, test_sentiments):
    lr_model_execution = lr_model(countvect_reviews_train, countvect_reviews_test, train_sentiments, tfidf_vect_reviews_train,
                                  tfidf_vect_reviews_test)
    lr_bow_predict = lr_model_execution.get('lr_bow_predict')
    lr_tf_idf_predict = lr_model_execution.get('lr_tf_idf_predict')

    lr_accuracy(test_sentiments, lr_bow_predict, lr_tf_idf_predict)

    lr_bow_score, lr_tf_idf_score = lr_give_accuracy(test_sentiments, lr_bow_predict, lr_tf_idf_predict)

    lr_classification_report(test_sentiments, lr_bow_predict, lr_tf_idf_predict)

    lr_confusion_matrix(test_sentiments, lr_bow_predict, lr_tf_idf_predict)

    return lr_bow_score, lr_tf_idf_score






