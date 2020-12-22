import pandas as pd # data processing
import numpy as np # working with arrays
from sklearn.model_selection import train_test_split # splitting the data
from sklearn.linear_model import LogisticRegression # model algorithm
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.metrics import jaccard_score as jss # evaluation metric
from sklearn.metrics import precision_score # evaluation metric
from sklearn.metrics import classification_report # evaluation metric
from sklearn.metrics import confusion_matrix # evaluation metric
from sklearn.metrics import log_loss # evaluation metric
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def predict(req_data):
    print(req_data)
    result = runJob()
    return result


def runJob():
    df= pd.read_csv('two.csv')
    # Importing and cleaning the data
    cols_names=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21],[22],[23],[24],[25],[26],[27],[28],[29],[30],[31],[32],[33],[34],[35],[36],[37],[38],[39],[40],[41],[42],[43],[44],[45],[46],[47],[48],[49],[50],[51],[52],[53],[54],[55],[56],[57],[58],[59],[60],[61],[62],[63],[64],[65],[66],[67],[68],[69],[70],[71],[72],[73],[74],[75],[76],[77],[78],[79],[80],[81],[82],[83],[84],[85],[86],[87],[88],[89],[90],[91],[92],[93],[94],[95],[96],[97],[98],[99],[100],[101],[102],[103],[104],[105],[106],[107],[108],[109],[110],[111],[112],[113],[114],[115],[116],[117],[118],[119],[120],[121],[122],[123],[124],[125],[126],[127],[128],[129],[130],[131],[132],[133],[134],[135],[136],[137],[138],[139],[140],[141],[142],[143],[144],[145],[146],[147],[148],[149],[150],[151],[152],[153],[154],[155],[156],[157],[158],[159],[160],[161],[162],[163],[164],[165],[166],[167],[168],[169],[170],[171],[172],[173],[174],[175],[176],[177],[178],[179],[180],[181],[182],[183],[184],[185],[186],[187],[188],[189],[190],[191],[192],[193],[194]]
    
    feature_cols = [[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[4],[5],[3]]
    
    X = np.asarray(cols_names)
    y = np.asarray(feature_cols)
    #print('X samples : ', X[:2])
    #print('y samples : ', y[:2])
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size = 0.2)
    ##print('X_train samples : ', X_train[:5])
    ##print('X_test samples : ', X_test[:5])
    ##print('y_train samples : ', y_train[:10])
    ##print('y_test samples : ', y_test[:10])
    # Modelling
    
    X_scaled = preprocessing.scale(X_train)
    ##print("X_scaled=", X_scaled [:3])
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    ##print(lr)
    # Predictions
    yhat = lr.predict(X_test)
    yhat_prob = lr.predict_proba(X_test)
    #print('yhat samples : ', yhat[:10])
    #print('yhat_prob samples : ',yhat_prob[:10])
    
    # 1. Jaccard Index
    #print('similerity index = ',jss(y_test, yhat, average='macro'))
    # 2. Precision Score
    #print('precision_score = ',precision_score(y_test, yhat, average='micro'))
    # 3. Log loss
    #print('log_loss = ', log_loss(["spam", "ham", "ham", "spam"],[[.1, .9], [.9, .1], [.8, .2], [.35, .65]]))
    # 4. Classificaton report
    target_names = ['exposed', 'diseased', 'susceptible' ,'recovred possible injury again','recovred', 'infectious' ]
    #print(classification_report(y_test, yhat, labels=[1, 2, 3,4,5,6], target_names=target_names))
    # 5. Confusion matrix
    predictions = lr.predict(X_test)
    cm = metrics.confusion_matrix(y_test, predictions)
    #print(cm)
    
    Classification1 = (classification_report(y_test, yhat, labels=[1], target_names = ['exposed'], zero_division=0))
    #print(Classification1)
    
    Classification2 = (classification_report(y_test, yhat, labels=[2], target_names = ['diseased'], zero_division=0))
    #print(Classification2)
    
    Classification3 = (classification_report(y_test, yhat, labels=[3], target_names = ['susceptible'], zero_division=0))
    #print(Classification3)
    
    Classification4 = (classification_report(y_test, yhat, labels=[4], target_names = ['recovred possible injury again'], zero_division=0))
    #print(Classification4)
    
    Classification5 = (classification_report(y_test, yhat, labels=[5], target_names = ['recovred'], zero_division=0))
    #print(Classification5)
    
    Classification6 = (classification_report(y_test, yhat, labels=[6], target_names = ['infectious'], zero_division=0))
    #print(Classification6)
    
    #print('lable [1] metric :')
    precision_metric = precision_score(y_test, yhat, average = "weighted", labels=[1], zero_division=0)
    #print('precion exposed',precision_metric)
    
    recall_metric = recall_score(y_test, yhat, average = "weighted", labels=[1], zero_division=0)
    #print('recall exposed', recall_metric)
    
    f1 = f1_score (y_test, yhat,average='weighted', labels=[1], zero_division=0)
    #print ('f1 score =', f1)
    
    #print('---------------------------------------------------------------')
    #print('lable [2] metric :')
    
    precision_metric = precision_score(y_test, yhat, average = "weighted", labels=[2], zero_division=0)
    #print('precion diseased',precision_metric)
    
    recall_metric = recall_score(y_test, yhat, average = "weighted", labels=[2], zero_division=0)
    #print('recall diseased', recall_metric)
    
    f2 = f1_score (y_test, yhat,average='weighted', labels=[2], zero_division=0)
    #print ('f2 score =', f2)
    
    #print('---------------------------------------------------------------')
    #print('lable [3] metric :')
    
    precision_metric = precision_score(y_test, yhat, average = "weighted", labels=[3], zero_division=0)
    #print('precion susceptible',precision_metric)
    
    recall_metric = recall_score(y_test, yhat, average = "weighted", labels=[3], zero_division=0)
    #print('recall susceptible', recall_metric)
    
    f2 = f1_score (y_test, yhat,average='weighted', labels=[3], zero_division=0)
    #print ('f2 score =', f2)
    
    #print('---------------------------------------------------------------')
    #print('lable [4] metric :')
    
    precision_metric = precision_score(y_test, yhat, average = "weighted", labels=[4], zero_division=0)
    #print('precion recovred possible injury again',precision_metric)
    
    recall_metric = recall_score(y_test, yhat, average = "weighted", labels=[4], zero_division=0)
    #print('recall recovred possible injury again', recall_metric)
    
    f2 = f1_score (y_test, yhat,average='weighted', labels=[4], zero_division=0)
    #print ('f2 score =', f2)
    
    #print('---------------------------------------------------------------')
    #print('lable [5] metric :')
    
    precision_metric = precision_score(y_test, yhat, average = "weighted", labels=[5], zero_division=0)
    #print('precion recovred',precision_metric)
    
    recall_metric = recall_score(y_test, yhat, average = "weighted", labels=[5], zero_division=0)
    #print('recall recovred', recall_metric)
    
    f2 = f1_score (y_test, yhat,average='weighted', labels=[5], zero_division=0)
    #print ('f2 score =', f2)
    
    #print('---------------------------------------------------------------')
    #print('lable [6] metric :')
    
    precision_metric = precision_score(y_test, yhat, average = "weighted", labels=[6], zero_division=0)
    ##print('precion infectious',precision_metric)
    
    recall_metric = recall_score(y_test, yhat, average = "weighted", labels=[6], zero_division=0)
    ##print('recall infectious', recall_metric)
    
    f2 = f1_score (y_test, yhat,average='weighted', labels=[6], zero_division=0)
    ##print ('f2 score =', f2)
    
    
    #print('---------------------------------------------------------------')

    result = ''
    
    if (0 <=precision_score(y_test, yhat, average = "weighted", labels=[1], zero_division=0) <= 0.73) or (recall_score(y_test, yhat, average = "weighted", labels=[1], zero_division=0) == 0.71) or (0<=f1_score (y_test, yhat,average='weighted', labels=[1], zero_division=0)<= 0.72):
        print('exposed')
        result = 'exposed'
    elif (0 <=precision_score(y_test, yhat, average = "weighted", labels=[1], zero_division=0) <= 0.73) and (recall_score(y_test, yhat, average = "weighted", labels=[1], zero_division=0) == 0.71) and (0<=f1_score (y_test, yhat,average='weighted', labels=[1], zero_division=0)<= 0.72):
        print('exposed')
        result = 'exposed'
    elif (0 <=precision_score(y_test, yhat, average = "weighted", labels=[1], zero_division=0) <= 0.73) or (recall_score(y_test, yhat, average = "weighted", labels=[1], zero_division=0) == 0.71) and (0<=f1_score (y_test, yhat,average='weighted', labels=[1], zero_division=0)<= 0.72):
        print('exposed')
        result = 'exposed'
    elif (0 <=precision_score(y_test, yhat, average = "weighted", labels=[1], zero_division=0) <= 0.73) and (recall_score(y_test, yhat, average = "weighted", labels=[1], zero_division=0) == 0.71) or (0<=f1_score (y_test, yhat,average='weighted', labels=[1], zero_division=0)<= 0.72):
        print('exposed')
        result = 'exposed'
    
    
    
    elif (0.83 <=precision_score(y_test, yhat, average = "weighted", labels=[2], zero_division=0) <= 0.89) or (0<= recall_score(y_test, yhat, average = "weighted", labels=[2], zero_division=0) <= 0.70) or (0.73 <=f1_score (y_test, yhat,average='weighted', labels=[2], zero_division=0) <= 0.76):
        print('diseased')
        result = 'diseased'
    elif (0.83 <=precision_score(y_test, yhat, average = "weighted", labels=[2], zero_division=0) <= 0.89) and (0<= recall_score(y_test, yhat, average = "weighted", labels=[2], zero_division=0) <= 0.70) and (0.73 <=f1_score (y_test, yhat,average='weighted', labels=[2], zero_division=0) <= 0.76):
        print('diseased')
        result = 'diseased'
    elif (0.83 <=precision_score(y_test, yhat, average = "weighted", labels=[2], zero_division=0) <= 0.89) or (0<= recall_score(y_test, yhat, average = "weighted", labels=[2], zero_division=0) <= 0.70) and (0.73 <=f1_score (y_test, yhat,average='weighted', labels=[2], zero_division=0) <= 0.76):
        print('diseased')
        result = 'diseased'
    elif (0.83 <=precision_score(y_test, yhat, average = "weighted", labels=[2], zero_division=0) <= 0.89) and (0<= recall_score(y_test, yhat, average = "weighted", labels=[2], zero_division=0) <= 0.70) or (0.73 <=f1_score (y_test, yhat,average='weighted', labels=[2], zero_division=0) <= 0.76):
        print('diseased')
        result = 'diseased'
    
    
    elif (0.74 <=precision_score(y_test, yhat, average = "weighted",labels=[3], zero_division=0) <= 0.82) and (0.72 <=recall_score(y_test, yhat, average = "weighted",labels=[3], zero_division=0) == 0.79) and (0.77 <=f1_score (y_test, yhat,average='weighted',labels=[3], zero_division=0) <= 0.94):
        print('susceptible')
        result = 'susceptible'
    elif (0.74 <=precision_score(y_test, yhat, average = "weighted",labels=[3], zero_division=0) <= 0.82) or (0.72 <=recall_score(y_test, yhat, average = "weighted",labels=[3], zero_division=0) == 0.79) or (0.77 <=f1_score (y_test, yhat,average='weighted',labels=[3], zero_division=0) <= 0.94):
        print('susceptible')
        result = 'susceptible'
    elif (0.74 <=precision_score(y_test, yhat, average = "weighted",labels=[3], zero_division=0) <= 0.82) or (0.72 <=recall_score(y_test, yhat, average = "weighted",labels=[3], zero_division=0) == 0.79) and (0.77 <=f1_score (y_test, yhat,average='weighted',labels=[3], zero_division=0) <= 0.94):
        print('susceptible')
        result = 'susceptible'
    elif (0.74 <=precision_score(y_test, yhat, average = "weighted",labels=[3], zero_division=0) <= 0.82) and (0.72 <=recall_score(y_test, yhat, average = "weighted",labels=[3], zero_division=0) == 0.79) or (0.77 <=f1_score (y_test, yhat,average='weighted',labels=[3], zero_division=0) <= 0.94):
        print('susceptible')
        result = 'susceptible'
    
    
    elif (precision_score(y_test, yhat, average = "weighted", labels=[4], zero_division=0) ==1) and (0.80 <=recall_score(y_test, yhat, average = "weighted", labels=[4], zero_division=0) <= 0.99) and (f1_score (y_test, yhat,average='weighted', labels=[4], zero_division=0) == 1):
        print('infectious')
        result = 'infectious'
    elif (precision_score(y_test, yhat, average = "weighted", labels=[4], zero_division=0) ==1) or (0.80 <=recall_score(y_test, yhat, average = "weighted", labels=[4], zero_division=0) <= 0.99) or (f1_score (y_test, yhat,average='weighted', labels=[4], zero_division=0) == 1):
        print('infectious')
        result = 'infectious'
    elif (precision_score(y_test, yhat, average = "weighted", labels=[4], zero_division=0) ==1) or (0.80 <=recall_score(y_test, yhat, average = "weighted", labels=[4], zero_division=0) <= 0.99) and (f1_score (y_test, yhat,average='weighted', labels=[4], zero_division=0) == 1):
        print('infectious')
        result = 'infectious'
    elif (precision_score(y_test, yhat, average = "weighted", labels=[4], zero_division=0) ==1) and (0.80 <=recall_score(y_test, yhat, average = "weighted", labels=[4], zero_division=0) <= 0.99) or (f1_score (y_test, yhat,average='weighted', labels=[4], zero_division=0) == 1):
        print('infectious')
        result = 'infectious'
    
    
    elif (precision_score(y_test, yhat, average = "weighted", labels=[5], zero_division=0) ==1) and (recall_score(y_test, yhat, average = "weighted", labels=[5], zero_division=0) == 1) and (0.98 <= f1_score (y_test, yhat,average='weighted', labels=[5], zero_division=0) <= 0.99):
        print('recovred')
        result = 'recovred'
    elif (precision_score(y_test, yhat, average = "weighted", labels=[5], zero_division=0) ==1) or (recall_score(y_test, yhat, average = "weighted", labels=[5], zero_division=0) == 1) or (0.98 <= f1_score (y_test, yhat,average='weighted', labels=[5], zero_division=0) <= 0.99):
        print('recovred')
        result = 'recovred'
    elif (precision_score(y_test, yhat, average = "weighted", labels=[5], zero_division=0) ==1) or (recall_score(y_test, yhat, average = "weighted", labels=[5], zero_division=0) == 1) and (0.98 <= f1_score (y_test, yhat,average='weighted', labels=[5], zero_division=0) <= 0.99):
        print('recovred')
        result = 'recovred'
    elif (precision_score(y_test, yhat, average = "weighted", labels=[5], zero_division=0) ==1) and (recall_score(y_test, yhat, average = "weighted", labels=[5], zero_division=0) == 1) or (0.98 <= f1_score (y_test, yhat,average='weighted', labels=[5], zero_division=0) <= 0.99):
        print('recovred')
        result = 'recovred'
    
    
    elif (0.90<= precision_score(y_test, yhat, average = "weighted", labels=[6], zero_division=0) <=0.99) and (recall_score(y_test, yhat, average = "weighted", labels=[6], zero_division=0) == 1) and   (0.95 <= f1_score (y_test, yhat,average='weighted', labels=[6], zero_division=0) <= 0.97):
        print('recovred possible injury again')
        result = 'recovred possible injury again'
    elif (0.90<= precision_score(y_test, yhat, average = "weighted", labels=[6], zero_division=0) <=0.99) or (recall_score(y_test, yhat, average = "weighted", labels=[6], zero_division=0) == 1) or   (0.95 <= f1_score (y_test, yhat,average='weighted', labels=[6], zero_division=0) <= 0.97):
        print('recovred possible injury again')
        result = 'recovred possible injury again'
    elif (0.90<= precision_score(y_test, yhat, average = "weighted", labels=[6], zero_division=0) <=0.99) or (recall_score(y_test, yhat, average = "weighted", labels=[6], zero_division=0) == 1) and   (0.95 <= f1_score (y_test, yhat,average='weighted', labels=[6], zero_division=0) <= 0.97):
        print('recovred possible injury again')
        result = 'recovred possible injury again'
    elif (0.90<= precision_score(y_test, yhat, average = "weighted", labels=[6], zero_division=0) <=0.99) and (recall_score(y_test, yhat, average = "weighted", labels=[6], zero_division=0) == 1) or   (0.95 <= f1_score (y_test, yhat,average='weighted', labels=[6], zero_division=0) <= 0.97):
        print('recovred possible injury again')
        result = 'recovred possible injury again'

    return result