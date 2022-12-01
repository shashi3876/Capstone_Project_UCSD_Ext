# Import libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)
# Load the model

save_path = 'model.h5'
## load tensorflow model
model = load_model(save_path)
#Load scalers for other features
main_scaler=pickle.load(open('mainScaler.pkl','rb'))
lag_num_tweets_scaler=pickle.load(open('lag_num_tweets_scaler.pkl','rb'))
lag_vd_sentiment_scaler=pickle.load(open('lag_vd_sentiment_scaler.pkl','rb'))
lag_replies_scaler=pickle.load(open('lag_replies_scaler.pkl','rb'))


@app.route('/')
def home_endpoint():
    return 'Welcome to BTC price prediction. PLease call api method with the required features!'

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Extract indivsual features
    X_test = np.array(data['last_60_days'])
    num_tweets = np.array(data['num_tweets'])
    vd_sentiment = np.array(data['vd_sentiment'])
    replies = np.array(data['replies'])
    
    #Do scaling
    X_test=main_scaler.transform(X_test.reshape(-1, 1))
    X_test = np.reshape(X_test, (1,60,1))    
    num_tweets=lag_num_tweets_scaler.transform(num_tweets.reshape(-1, 1))
    vd_sentiment=lag_vd_sentiment_scaler.transform(vd_sentiment.reshape(-1, 1))
    replies=lag_replies_scaler.transform(replies.reshape(-1, 1))
    X_test=np.append(X_test,[num_tweets,vd_sentiment,replies])
    X_test = np.reshape(X_test, (1,63,1))
    
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(X_test)
    #Inverse transform
    ans=main_scaler.inverse_transform(prediction[0][0].reshape(1, -1))
    return jsonify(f'Predicted BTC price is {str(ans[0][0])} USD.')

if __name__ == '__main__':
    app.run(port=5000, debug=False)