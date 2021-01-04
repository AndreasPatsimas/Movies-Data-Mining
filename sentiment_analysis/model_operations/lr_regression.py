from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def lr_model(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews):
    #training the model
    lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)

    #Fitting the model for Bag of words
    lr_bow = lr.fit(cv_train_reviews, train_sentiments)
    print(lr_bow)

    #Fitting the model for tfidf features
    lr_tf_idf = lr.fit(tv_train_reviews, train_sentiments)
    print(lr_tf_idf)

    #Predicting the model for bag of words
    lr_bow_predict = lr.predict(cv_test_reviews)
    print(lr_bow_predict)

    ##Predicting the model for tfidf features
    lr_tf_idf_predict = lr.predict(tv_test_reviews)
    print(lr_tf_idf_predict)

    my_dictionary = {'lr_bow_predict': lr_bow_predict, 'lr_tf_idf_predict': lr_tf_idf_predict}

    return my_dictionary

def lr_accuracy(test_sentiments, lr_bow_predict, lr_tf_idf_predict):
    # Accuracy score for bag of words
    lr_bow_score = accuracy_score(test_sentiments, lr_bow_predict)
    print("lr_bow_score :", lr_bow_score)
    # Accuracy score for tfidf features
    lr_tf_idf_score = accuracy_score(test_sentiments, lr_tf_idf_predict)
    print("lr_tfidf_score :", lr_tf_idf_score)

def lr_classification_report(test_sentiments, lr_bow_predict, lr_tf_idf_predict):
    # Classification report for bag of words
    lr_bow_report = classification_report(test_sentiments, lr_bow_predict, target_names=['Positive', 'Negative'])
    print(lr_bow_report)

    # Classification report for tfidf features
    lr_tfidf_report = classification_report(test_sentiments, lr_tf_idf_predict, target_names=['Positive', 'Negative'])
    print(lr_tfidf_report)

def lr_confusion_matrix(test_sentiments, lr_bow_predict, lr_tf_idf_predict):
    # confusion matrix for bag of words
    cm_bow = confusion_matrix(test_sentiments, lr_bow_predict, labels=[1, 0])
    print(cm_bow)
    # confusion matrix for tfidf features
    cm_tfidf = confusion_matrix(test_sentiments, lr_tf_idf_predict, labels=[1, 0])
    print(cm_tfidf)

def lr_regression_execution(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews, tv_test_reviews, test_sentiments):
    lr_model_execution = lr_model(cv_train_reviews, cv_test_reviews, train_sentiments, tv_train_reviews,
                                  tv_test_reviews)
    lr_bow_predict = lr_model_execution.get('lr_bow_predict')
    lr_tf_idf_predict = lr_model_execution.get('lr_tf_idf_predict')

    lr_accuracy(test_sentiments, lr_bow_predict, lr_tf_idf_predict)

    lr_classification_report(test_sentiments, lr_bow_predict, lr_tf_idf_predict)

    lr_confusion_matrix(test_sentiments, lr_bow_predict, lr_tf_idf_predict)