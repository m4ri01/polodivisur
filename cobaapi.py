from flask import Flask,jsonify,request
from tqdm import tqdm
import numpy as np
import word2vec
import pickle
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from sqlalchemy import create_engine
import pymysql
import pandas as pd
import joblib
from math import radians, cos, sin, asin, sqrt



app = Flask(__name__)
w2vModel = Word2Vec.load("simpan")
kmeans = pickle.load(open('model.sav','rb'))
w2vKata = pickle.load(open('w2vKata.pkl','rb'))
db_connection_str = 'mysql+pymysql://xxxxxx:xxxxxx;@127.0.0.1/polodivisur'
db_connection = create_engine(db_connection_str)
df = pd.read_sql('SELECT * FROM rs',con=db_connection)
longRS = df.longitude.to_list()
latRS = df.latitude.to_list()

def haversine(lon1, lat1, lon2, lat2): 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

@app.route('/',methods=['GET','POST'])
def index():
    if(request.method=='POST'):
        reqjson = request.get_json()
        return jsonify({'request':"kosong"}),201
    else:
        return jsonify({'request':'kosong'})

@app.route('/rs/<string:longlat>',methods=['GET'])
def hitlonglat(longlat):
    global longRS
    global latRS
    longlatUser = longlat.split(",")
    longUser = float(longlatUser[0])
    latUser = float(longlatUser[1])
    iduser = longlatUser[2]
    jarak = []
    for i in range(len(longRS)):
        hitjarak = haversine(longUser,latUser,longRS[i],latRS[i])
        jarak.append(hitjarak)
    rumahSakit = jarak.index(min(jarak))
    rumahSakit = int(rumahSakit)+1
    return jsonify({'rsid':rumahSakit,"id":iduser})



@app.route('/nlp/<string:bahasa>',methods=['GET'])
def proses(bahasa):
    global w2vKata
    global w2vModel
    global kmeans
    selectid = bahasa.split(",")
    id = int(selectid[-1])
    del selectid[-1]
    bahasa = ",".join(selectid)
    size = 8
    katainput = [bahasa]
    sent_vectors = []; 
    count_error = 0
    for review in tqdm(katainput): 
        sent_vec = np.zeros(size)
        cnt_words =0; 
        for word in review.split(","):
            if word in w2vKata:
                vec = w2vModel.wv[word]
                sent_vec += vec
                cnt_words += 1
        if cnt_words != 0:
            sent_vec /= cnt_words
        sent_vectors.append(sent_vec)
    inputan = np.array(sent_vectors)
    prediksi = kmeans.predict(inputan)
    kirim = int(prediksi[0])
    print(prediksi)
    return jsonify({'result':kirim,"id":id})

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')
