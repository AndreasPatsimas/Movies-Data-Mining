import pandas as pd

def import_data_csv():
    #import training data
    my_data=pd.read_csv('C:/Users/andre/Desktop/ΣΠΟΥΔΕΣ/ΠΜΣ ΠΑΠΕΙ/ΜΑΘΗΜΑΤΑ/DATA MINING/xalkidh/ergasia/input/IMDB Dataset.csv')
    print(my_data.shape)
    print(my_data.head(10))

    #Summary of the dataset
    print(my_data.describe())

    #sentiment count
    print(my_data['sentiment'].value_counts())

    return my_data