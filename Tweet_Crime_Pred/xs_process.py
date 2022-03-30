import os
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


stemmer = PorterStemmer()

# NLTK Stemming
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


# NLTK tokenization
def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def initEnhanced(tweet):
    my_list = []
    my_list_label = []
    my_list_mat_scores = {}

    directory = r'data/groups'
    for filename in os.listdir(directory):    
        if filename.endswith("testTweetIDs"):
            print('')
        else:
            fileTo = os.path.join(directory, filename)
            #print(fileTo)
            spt = [z.strip() for z in fileTo.split('\\')]
            ext = spt[1]
            f = open(fileTo, encoding="utf-8")
            line = f.readline()        
            c = 0
            while line:
                u=line
                u=u.encode('unicode-escape').decode('utf-8')
                #print(u)
                my_list.append(u)
                my_list_label.append(ext)
                line = f.readline()
                c = c+1            
            f.close()

    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    c = 0
    for x in range(len(my_list)):
        #print(my_list[x]+"----"+my_list_label[x])    
        c = c+1
        key = my_list_label[x]+"_"+str(c)
        # this can take some time    
        tfs_matrix = tfidf.fit_transform([my_list[x], tweet])
        result_array = cosine_similarity(tfs_matrix[0:1], tfs_matrix[1:])  # the first element of tfs_matrix is matched with other elements
        score = result_array[0][0]
        my_list_mat_scores[key] = score    

    sort_orders = sorted(my_list_mat_scores.items(), key=lambda x: x[1], reverse=True)

    rep = "predicted_results/rep"
    file1 = open(rep,"w")
    for i in sort_orders:
            #print(i[0], i[1])
            spt = [z.strip() for z in i[0].split('_')]
            detection = spt[0]
            file1.write('System : Enhancement')
            file1.write('\nTweet : '+tweet)
            if i[1]>0.3:
                file1.write('\nCrime Category : '+detection)
                #file1.write('\nScore : '+str(i[1]))
            else:
                file1.write('\nCrime Category : None(Normal Tweet)')
                #file1.write('\nScore : '+str(i[1]))
            break

    file1.close() #to change file access modes
    os.system('notepad.exe '+rep)

def plotterAccuracy():
    #graph
    p1 = []
    p2 = []
    with open("predicted_results/estats.txt", encoding="utf-8") as f:
        cnt = 0
        for line in f:
            cnt = cnt+1
            result = [line.strip() for line in line.split(':')]
            if cnt == 1:            
                p1.append(float(result[1]))
            if cnt == 2:            
                p2.append(float(result[1]))        
            
    with open("predicted_results/pstats.txt", encoding="utf-8") as f:
        cnt = 0
        for line in f:
            cnt = cnt+1
            result = [line.strip() for line in line.split(':')]
            if cnt == 1:            
                p1.append(float(result[1]))
            if cnt == 2:            
                p2.append(float(result[1]))        

    with open("predicted_results/xstats.txt", encoding="utf-8") as f:
        cnt = 0
        for line in f:
            cnt = cnt+1
            result = [line.strip() for line in line.split(':')]
            if cnt == 1:            
                p1.append(float(result[1]))
            if cnt == 2:            
                p2.append(float(result[1]))       


    ###########################

    my_listLabels = []
    my_list = []

    my_listLabels.append('TF-IDF + KNN')            
    my_list.append(p1[0])
    my_listLabels.append('TF-IDF + SVM')            
    my_list.append(p1[1])
    my_listLabels.append('Jaccard + DNN')            
    my_list.append(p1[2])

    # Plot the bar graph
    plot = plt.bar(my_listLabels,my_list)
    plot[0].set_color('red')
    plot[1].set_color('blue')
    plot[2].set_color('orange')
     
    # Add the data value on head of the bar
    for value in plot:
        height = value.get_height()    
        plt.text(value.get_x() + value.get_width()/2.,1.002*height,'%f' % float(height), ha='center', va='bottom')
     
    # Add labels and title
    plt.title("Tweet Crime Sensivity Prediction Accuracy")
    plt.xlabel("Classifier")
    plt.ylabel("Score")
     
    # Display the graph on the screen
    plt.show()

    #############################

    
def plotterDuration():
    #graph
    p1 = []
    p2 = []
    with open("predicted_results/estats.txt", encoding="utf-8") as f:
        cnt = 0
        for line in f:
            cnt = cnt+1
            result = [line.strip() for line in line.split(':')]
            if cnt == 1:            
                p1.append(float(result[1]))
            if cnt == 2:            
                p2.append(float(result[1]))        
            
    with open("predicted_results/pstats.txt", encoding="utf-8") as f:
        cnt = 0
        for line in f:
            cnt = cnt+1
            result = [line.strip() for line in line.split(':')]
            if cnt == 1:            
                p1.append(float(result[1]))
            if cnt == 2:            
                p2.append(float(result[1]))        

    with open("predicted_results/xstats.txt", encoding="utf-8") as f:
        cnt = 0
        for line in f:
            cnt = cnt+1
            result = [line.strip() for line in line.split(':')]
            if cnt == 1:            
                p1.append(float(result[1]))
            if cnt == 2:            
                p2.append(float(result[1]))       

   

    ###########################

    my_listLabels = []
    my_list = []

    my_listLabels.append('TF-IDF + KNN')            
    my_list.append(p2[0])
    my_listLabels.append('TF-IDF + SVM')            
    my_list.append(p2[1])
    my_listLabels.append('Jaccard + DNN')            
    my_list.append(p2[2])

    # Plot the bar graph
    plot = plt.bar(my_listLabels,my_list)
    plot[0].set_color('red')
    plot[1].set_color('blue')
    plot[2].set_color('orange')
     
    # Add the data value on head of the bar
    for value in plot:
        height = value.get_height()    
        plt.text(value.get_x() + value.get_width()/2.,1.002*height,'%f' % float(height), ha='center', va='bottom')
     
    # Add labels and title
    plt.title("Tweet Crime Sensivity Prediction Duration")
    plt.xlabel("Classifier")
    plt.ylabel("Duration")
     
    # Display the graph on the screen
    plt.show()

    #############################

