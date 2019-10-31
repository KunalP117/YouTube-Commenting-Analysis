import datetime                                                                 
import code #Allows interaction with Python code                                
#import mysql.connector #MySQL Connection
import urllib.request #Collect YouTube Thumbnails using URL                     
import pickle #Import/Export Large Lists                                        
import time


#Channel Information Class
class ChanInfo:
    def __init__(self,channelid=None,published=None,language=None,videocount=None,viewcount=None,subscount=None,commentcount=None,crawltime=None):
        #a if cond else b
        #Static Channel Information
        self.channelid=[] if channelid is None else channelid
        self.published=[] if published is None else published
        self.language=[] if language is None else language
        self.videocount=[] if videocount is None else videocount
        self.viewcount=[] if viewcount is None else viewcount
        self.subscount=[] if subscount is None else subscount
        self.commentcount=[] if commentcount is None else commentcount
        self.crawltime=[] if crawltime is None else crawltime
        #Daily Channel Information
        self.day=[]
        self.views=[]
        self.comments=[]
        self.likes=[]
        self.dislikes=[]
        self.shares=[]
        self.subsgained=[]
        self.subslost=[]

    #Add Data to Class (ytOverviewChannel)
    def adddata(self,day,views,comments,likes,dislikes,shares,subsgained,subslost):
        self.day.append(day)
        self.views.append(views)
        self.comments.append(comments)
        self.likes.append(likes)
        self.dislikes.append(dislikes)
        self.shares.append(shares)
        self.subsgained.append(subsgained)
        self.subslost.append(subslost)

#Video Information Class
class VidInfo:
    def __init__(self,nid=None,cid=None,published=None,channelid=None,language=None,viewcount=None,likecount=None,dislikecount=None,commentcount=None,crawltime=None,duration=None,category=None):
        #a if cond else b
        self.nid=[] if nid is None else nid
        self.cid=[] if cid is None else cid
        #Static Video Information
        self.published=[] if published is None else published
        self.channelid=[] if channelid is None else channelid
        self.language=[] if language is None else language
        self.viewcount=[] if viewcount is None else viewcount
        self.likecount=[] if likecount is None else likecount
        self.dislikecount=[] if dislikecount is None else dislikecount
        self.commentcount=[] if commentcount is None else commentcount
        self.crawltime=[] if crawltime is None else crawltime
        self.duration=[] if duration is None else duration
        self.category=[] if category is None else category
        #Daily Video Information
        self.day=[]
        self.views=[]
        self.comments=[]
        self.likes=[]
        self.dislikes=[]
        self.shares=[]
        self.subsgained=[]
        self.subslost=[]
        self.favadd=[]
        self.favlost=[]

    #Add Data to Class (ytOverviewVideo)
    def addviews(self,day,views,comments,likes,dislikes,shares,subsgained,subslost,favadd,favlost):
        self.day.append(day)
        self.views.append(views)
        self.comments.append(comments)
        self.likes.append(likes)
        self.dislikes.append(dislikes)
        self.shares.append(shares)
        self.subsgained.append(subsgained)
        self.subslost.append(subslost)
        self.favadd.append(favadd)
        self.favlost.append(favlost)

#Title Information Class
class VidTitle:
    def __init__(self,nid=None,cid=None,channelid=None,crawltime=None,title=None):
        self.nid=[] if nid is None else nid
        self.cid=[] if cid is None else cid
        self.channelid=[] if  channelid is None else channelid
        self.crawltime=[] if crawltime is None else crawltime
        self.title=[] if title is None else title
        #Title Optimization Information
        self.time=[]
        self.oldtitle=[]
        self.newtitle=[]

    #Add Data to Class (titleOptimizationLog)
    def addtitle(self,time,oldtitle,newtitle):
        self.time.append(time)
        self.oldtitle.append(oldtitle)
        self.newtitle.append(newtitle)

#Thumbnail Information Class
class VidThumb:
    def __init__(self,nid=None,cid=None,channelid=None):
        self.nid=[] if nid is None else nid
        self.cid=[] if cid is None else cid
        self.channelid=[] if  channelid is None else channelid
        #Video Optimization Information
        self.time=[]
        self.url=[]

    def addthumb(self,time,url):
        self.time.append(time)
        self.url.append(url)

#Auxiliary Channel Information Class
class ChanAuxInfo:
    def __init__(self,channelid=None,published=None,language=None,videocount=None,viewcount=None,subscount=None,commentcount=None,crawltime=None):
        #a if cond else b
        #Static Channel Information
        self.channelid=[] if channelid is None else channelid
        #Daily Video Information
        self.day=[]
        self.subscribers=[]
        self.videos=[]
        self.views=[]
        self.comments=[]

    #Add Data to Class (ytOverviewChannel)
    def adddata(self,day,subscribers,videos,views,comments):
        self.day.append(day)
        self.subscribers.append(subscribers)
        self.videos.append(videos)
        self.views.append(views)
        self.comments.append(comments)

#Data Class
class Data:
    def __init__(self,videoNumFeatures=None,videoLabels=None,title=None,image=None):
        self.videoNumFeatures=[] if videoNumFeatures is None else videoNumFeatures
        self.videoLabels=[] if videoLabels is None else videoLabels
        self.title=[] if title is None else title
        self.image=[] if image is None else image
###########################################################################################
###########################################################################################
#Save/Load Large Variables in
def saveloadList(lst,fileName,SL):
    if(SL == 1): #Save List
        with open(fileName,'wb') as pickle_file:
            pickle.dump(lst, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        return
    else: #Load List
        with open(fileName, 'rb') as pickle_load:
            lst = pickle.load(pickle_load)
        return lst
###############################################################################################
#Get Dynamic Channel Information
def gatherDynChanInfo(cnx,X):
    cursor = cnx.cursor()
    cursor.execute('SELECT day,views,comments,likes,dislikes,shares,subscribersGained,subscribersLost FROM ytOverviewChannel WHERE channelId=\'{}\''.format(X.channelid))
    channelData1 = cursor.fetchall()
    for k in range(len(channelData1)):
        day=time.mktime(list(channelData1)[k][0].timetuple())
        views=list(channelData1)[k][1]
        comments=list(channelData1)[k][2]
        likes=list(channelData1)[k][3]
        dislikes=list(channelData1)[k][4]
        shares=list(channelData1)[k][5]
        subsgained=list(channelData1)[k][6]
        subslost=list(channelData1)[k][7]
        X.adddata(day,views,comments,likes,dislikes,shares,subsgained,subslost)
    return X

#Get Channel Information
def gatherChanInfo(cnx,channelid):
    cursor = cnx.cursor()
    cursor.execute('SELECT published,language, videoCount, viewCount, subscriberCount, commentCount, crawlTime  FROM ytChannel WHERE channelId={}'.format(str(channelid)))
    channelData = cursor.fetchall()
    published=list(channelData)[0][0].timestamp()
    language=list(channelData)[0][1]
    videocount=list(channelData)[0][2]
    viewcount=list(channelData)[0][3]
    subscount=list(channelData)[0][4]
    commentcount=list(channelData)[0][5]
    crawltime=[] if list(channelData)[0][6] is None else list(channelData)[0][6].timestamp()
    X=ChanInfo(channelid,published,language,videocount,viewcount,subscount,commentcount,crawltime)
    #code.interact(local=locals())
    #Gather Dynamics
    X=gatherDynChanInfo(cnx,X)
    return X

###############################################################################################
#Find All Unqiue Videos
def constructUniqueViews(cnx):
    cursor = cnx.cursor()
    cursor.execute('select distinct(videoYid) from ytVideo')
    lst = cursor.fetchall()
    #Gather desired information
    videoUnique = []
    for k in range(len(lst)):
        videoUnique.append(lst[k][0])
    saveloadList(videoUnique,'videoUnique.pkl',1)
    return

#Find All Videos in Channel
def getVideoInChan(cnx,channelId):
    cursor = cnx.cursor()
    cursor.execute('SELECT DISTINCT(videoYid) FROM ytVideo WHERE channelId=\'{}\''.format(channelId))
    lst = cursor.fetchall()
    #Gather desired information
    videoUnique = []
    for k in range(len(lst)):
        videoUnique.append(lst[k][0])
    return videoUnique

#Get Dynamic Video Information
def gatherDynVidInfo(cnx,X):
    cursor = cnx.cursor()
    for tbl in range(1,11):
        cursor.execute('SELECT day,views,comments,likes,dislikes,shares,subscribersGained,subscribersLost,favoritesAdded,favoritesRemoved FROM ytOverviewVideo{} WHERE videoId=\'{}\''.format(tbl,X.nid))
        videoData1 = cursor.fetchall()
        for k in range(len(videoData1)):
            day=time.mktime(list(videoData1)[k][0].timetuple())
            views=list(videoData1)[k][1]
            comments=list(videoData1)[k][2]
            likes=list(videoData1)[k][3]
            dislikes=list(videoData1)[k][4]
            shares=list(videoData1)[k][5]
            subsgained=list(videoData1)[k][6]
            subslost=list(videoData1)[k][7]
            favadd=list(videoData1)[k][8]
            favlost=list(videoData1)[k][9]
            X.addviews(day,views,comments,likes,dislikes,shares,subsgained,subslost,favadd,favlost)
    return X



#Get Video Information
def gatherVidInfo(cnx,videoYid):
    cursor = cnx.cursor()
    cursor.execute('SELECT videoId,published,channelId,viewCount,likeCount,dislikeCount,commentCount,crawlTime,duration,categoryId FROM ytVideo WHERE videoYid=\'{}\''.format(videoYid))
    videoData = cursor.fetchall()
    nid=list(videoData)[0][0]
    cid=videoYid
    published=list(videoData)[0][1].timestamp()
    channelid=list(videoData)[0][2]
    viewcount=list(videoData)[0][3]
    likecount=list(videoData)[0][4]
    dislikecount=list(videoData)[0][5]
    commentcount=list(videoData)[0][6]
    crawltime=[] if list(videoData)[0][7] is None else list(videoData)[0][7].timestamp()
    duration=list(videoData)[0][8]
    category=list(videoData)[0][9]
    cursor.execute('SELECT language FROM ytChannel WHERE channelId={}'.format(list(videoData)[0][2]))
    language = cursor.fetchone()
    X=VidInfo(nid,cid,published,channelid,language[0],viewcount,likecount,dislikecount,commentcount,crawltime,duration,category)
    #Gather Dynamics
    X=gatherDynVidInfo(cnx,X)
    return X

###############################################################################################
#Get Title Optimization
def gatherTitOptimizationInfo(cnx,X):
    cursor = cnx.cursor()
    cursor.execute('SELECT oldTitle,newTitle,timestamp FROM titleOptimizationLog WHERE videoId=\'{}\''.format(X.cid))
    videoData1 = cursor.fetchall()
    for k in range(len(videoData1)):
       oldtitle=list(videoData1)[k][0]
       newtitle=list(videoData1)[k][1]
       time=list(videoData1)[k][2].timestamp()
       X.addtitle(time,oldtitle,newtitle)
    return X

#Get Title Information
def gatherVidTitle(cnx,videoYid):
    cursor = cnx.cursor()
    cursor.execute('SELECT videoId,published,channelId,title,crawlTime FROM ytVideo WHERE videoYid=\'{}\''.format(videoYid))
    videoData = cursor.fetchall()
    nid=list(videoData)[0][0]
    cid=videoYid
    published=list(videoData)[0][1].timestamp()
    channelid=list(videoData)[0][2]
    title=list(videoData)[0][3]
    crawltime=[] if list(videoData)[0][4] is None else list(videoData)[0][4].timestamp()
    X=VidTitle(nid,cid,channelid,crawltime,title)
    #Gather Optimization Information
    X=gatherTitOptimizationInfo(cnx,X)
    return X

###############################################################################################
#Get Thumb Information
def gatherThumbOptimizationInfo(cnx,X):
    cursor = cnx.cursor()
    cursor.execute('SELECT url,created FROM vpicksThumbnails WHERE videoId=\'{}\''.format(X.cid))
    videoData1 = cursor.fetchall()
    for k in range(len(videoData1)):
       url='http:{}'.format(list(videoData1)[k][0])
       time=list(videoData1)[k][1].timestamp()
       X.addthumb(time,url)
    return X

#Get Thumb Information
def gatherVidThumb(cnx,videoYid,nid,channelid):
    cursor = cnx.cursor()
    cursor.execute('SELECT url,created FROM vpicksOptimisationLog WHERE videoId=\'{}\''.format(videoYid))
    videoData = cursor.fetchall()
    X=VidThumb(nid,videoYid,channelid)
    #for k in range(len(videoData)):
    #   url='http:{}'.format(list(videoData)[k][0])
    #   time=list(videoData)[k][1].timestamp()
    #   X.addthumb(time,url)
    #Gather Optimization Information
    X=gatherThumbOptimizationInfo(cnx,X)
    return X

###############################################################################################
def videoNumFeatures(C,V):
    #Get Time Index of Video Publication for Channel
    min_list=[abs(k-V.published) for k in C.day]
    day_index=min_list.index(min(min_list))
    #Construct Channel Features
    chantotalviews = sum([abs(number) for number in C.views[:day_index+1]])
    chantotalcomments = sum([abs(number) for number in C.comments[:day_index+1]])
    chantotallikes = sum([abs(number) for number in C.likes[:day_index+1]])
    chantotaldislikes = sum([abs(number) for number in C.dislikes[:day_index+1]])
    chantotalshares = sum([abs(number) for number in C.dislikes[:day_index+1]])
    chantotalsubsgained = sum([abs(number) for number in C.subsgained[:day_index+1]])
    chantotalsubslost = sum([abs(number) for number in C.subslost[:day_index+1]])
    chantotalsubscribers = max(chantotalsubsgained-chantotalsubslost,0)
    #Construct Video Features
    duration = V.duration
    category = V.category
    #Return videoFeature list
    return [duration,category,chantotalviews,chantotalcomments,chantotallikes,chantotaldislikes,chantotalshares,chantotalsubsgained,chantotalsubslost,chantotalsubscribers]



def videoLabels(V,deltaDays):
    infoTime=V.published+deltaDays*86400
    min_list=[abs(k-infoTime) for k in V.day]
    day_index=min_list.index(min(min_list))
    #Construct Labels
    totalviews=sum([abs(number) for number in V.views[:day_index+1]])
    totalcomments=sum([abs(number) for number in V.comments[:day_index+1]])
    totallikes=sum([abs(number) for number in V.likes[:day_index+1]])
    totaldislikes=sum([abs(number) for number in V.dislikes[:day_index+1]])
    totalshares=sum([abs(number) for number in V.shares[:day_index+1]])
    totalsubsgained=sum([abs(number) for number in V.subsgained[:day_index+1]])
    totalsubslost=sum([abs(number) for number in V.subslost[:day_index+1]])
    totalfavadd=sum([abs(number) for number in V.favadd[:day_index+1]])
    totalfavlost=sum([abs(number) for number in V.favlost[:day_index+1]])
    return [totalviews,totalcomments,totallikes,totaldislikes,totalshares,totalsubsgained,totalsubslost,totalfavadd,totalfavlost]

###############################################################################################
#Construct ChanAuxInfo
def gatherChanAuxInfo(C,vidlist):
    #Videos in Channel
    indexV=[k for k in range(len(vidlist)) if vidlist[k].channelid == C.channelid]
    X=ChanAuxInfo(C.channelid)
    #Gather Information for Each Time Interval
    for w in range(len(C.day)):
        print('\033[1;35;48m {0:5} \033[1;34;48m {1:5} \033[0m '.format(w,len(C.day)))
        day=C.day[w]
        daymin=C.day[w]
        daymax=C.day[w]+86400
        #Parameters
        subscribers=0
        videos=0
        views=0
        comments=0
        #Find all Videos with Information
        for k in range(len(indexV)):
            V=vidlist[k]
            indexTime=[m for m in range(len(V.day)) if V.day[m] >= daymin and V.day[m] < daymax]
            if len(indexTime) !=0:
                videos += 1
                subsgained=sum([abs(V.subsgained[m]) for m in indexTime])
                subslost=sum([abs(V.subslost[m]) for m in indexTime])
                subscribers += subsgained-subslost
                comments += sum([abs(V.comments[m]) for m in indexTime])
                views += sum([abs(V.views[m]) for m in indexTime])
        #Store Information
        X.adddata(day,subscribers,videos,views,comments)
    #Return Data
    return X    

###########################################################################################
###########################################################################################













