#getTheTable
#
#
#
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from sklearn.metrics import confusion_matrix


def getTokens(input):
    tokensByDot,allTokens,tokensByAmpersand,tokensByUnderscore, tokensByApostrophe =[], [], [],[],[]
    tokensBySemicolon, tokensByEqual, tokensByColon =[],[], []
    tokensByQuestion, tokensByPersentage=[],[]
    tokensBySlash = str(input.encode('utf-8')).split('/')
    for i in tokensBySlash:
        tmp = str(i).split('.')
        tokensByDot+=tmp
    for i in tokensByDot:
        tmp = str(i).split('&')
        tokensByAmpersand+=tmp
    for i in tokensByAmpersand:
        tmp = str(i).split('_')
        tokensByUnderscore+=tmp
    for i in tokensByUnderscore:
        tmp = str(i).split(';')
        tokensBySemicolon+=tmp
    for i in tokensBySemicolon:
        tmp = str(i).split('=')
        tokensByEqual+=tmp
    for i in tokensByEqual:
        tmp = str(i).split(':')
        tokensByColon+=tmp
    for i in tokensByColon:
        tmp = str(i).split("'")
        tokensByApostrophe+=tmp
    for i in tokensByApostrophe:
        tmp = str(i).split('?')
        tokensByQuestion+=tmp
    for i in tokensByQuestion:
        tmp = str(i).split('%')
        tokensByPersentage+=tmp
    for j in tokensByPersentage:
        tmp = str(j).split('-')
        allTokens+=tmp    
    if 'com' in allTokens:
        allTokens.remove('com')
    elif 'b' in allTokens:
        allTokens.remove('b')
    elif 'www' in allTokens:
        allTokens.remove('www')
    elif '' in allTokens:
        allTokens.remove('')
    allTokens = list(set(allTokens))
    return allTokens
'''
def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')
    allTokens = []
    for i in tokensBySlash:
        tokens=str(i).split('-')
        tokensByDot = []
        for j in range(0, len(tokens)):
            tempTokens = str(tokens[j]).split('.')
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))
    if 'com' in allTokens:
        allTokens.remove('com')
    return allTokens
  '''  

def getData():
    allurls = '/Users/YumeiAdmin/Desktop/research/data_updated.txt'	#path to our all urls file
    allurlscsv = pd.read_csv(allurls,',',error_bad_lines=False)	#reading file
    allurlsdata = pd.DataFrame(allurlscsv)	#converting to a dataframe
    allurlsdata = np.array(allurlsdata)	#converting it into an array
    random.shuffle(allurlsdata)	#shuffling
    y = [d[1] for d in allurlsdata]	#all labels 
    corpus = [d[0] for d in allurlsdata]	#all urls corresponding to a label (either good or bad)
    
    blacklistPath = '/Users/YumeiAdmin/Desktop/research/blacklist.txt'
    blacklistCSV = pd.read_csv(blacklistPath, ',' , error_bad_lines = False)
    blacklistData = pd.DataFrame(blacklistCSV)
    blacklistData = np.array(blacklistData)
    blacklist = [d[2] for d in blacklistData]
    return corpus,y, blacklist

def printResult(cm, results,total):
    print("Accuracy: " , (cm[0][0]+cm[1][1])/total)
    print('avg(10 fold) = ' ,np.mean(results))
    print("Misclassification rate: ", (cm[0][1]+cm[1][0])/total)
    print("TPR: ", cm[1][1]/(cm[1][1]+cm[0][1]))
    print("FPR: " , cm[1][1]/(cm[0][0]+cm[0][1]))
    print("TNR: " , cm[0][0]/(cm[0][0]+cm[0][1]))
    print("Precision: " , cm[1][1]/(cm[0][1]+cm[1][1]))
    print("Prevalance: " ,(cm[1][0]+cm[1][1])/total)
    print("FalseNegative: " , cm[1][0]/total)
    

def main():
    corpus,y, blacklist = getData()
    vectorizer = TfidfVectorizer(tokenizer = getTokens, min_df = 2, max_df = .60 )	#get a vector for each url but use our customized tokenizer
    
    #X = vectorizer.fit_transform(corpus) #get the X vector
    X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.25, random_state=0)	#split into training and testing set 80/20 ratio
    #logistic regression
    X_train = vectorizer.fit_transform(X_train)
    svm = SVC(kernel = 'linear')
    svm.fit(X_train, y_train)
    #print(X_test[:5])
    #print(X_train[:1])
    print("----------------------------------------------------------")
    vec = vectorizer.transform(X_test)
    #print(vec[:1])
    y_pred =[]
    for i in range(len(X_test)):
        if X_test[i] in blacklist:
            y_pred.append(1)
        else:
            y_pred.append(int(svm.predict(vec[i])))
    
    y_pred = np.array(y_pred)
    #print(y_pred[:5])
    total = len(y_test)

    cm = confusion_matrix(y_test, y_pred)
    results = cross_val_score(estimator = svm, X= X_train, y=y_train, cv =10)
    printResult(cm, results, total)


main()
