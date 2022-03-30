import os
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer


def tfidf_dist(string1, string2):    
    set1 = set(string1)
    set2 = set(string2)
    return len(set1.intersection(set2))/min(len(set1), len(set2))

def initExisting(tweet):
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


    c = 0
    for x in range(len(my_list)):
        #print(my_list[x]+"----"+my_list_label[x])    
        c = c+1
        key = my_list_label[x]+"_"+str(c)
        # this can take some time    
        score = tfidf_dist(my_list[x], tweet)    
        my_list_mat_scores[key] = score    

    sort_orders = sorted(my_list_mat_scores.items(), key=lambda x: x[1], reverse=True)

    rep = "predicted_results/rep"
    file1 = open(rep,"w")
    for i in sort_orders:
            #print(i[0], i[1])
            spt = [z.strip() for z in i[0].split('_')]
            detection = spt[0]
            file1.write('System : Existing')
            file1.write('\nTweet : '+tweet)
            file1.write('\nCrime Category : '+detection)
            #file1.write('\nScore : '+str(i[1]))                
            break

    file1.close() #to change file access modes
    os.system('notepad.exe '+rep)   

