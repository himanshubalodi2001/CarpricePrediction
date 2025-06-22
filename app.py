# app.py
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# In‐memory user store
users = {}

# 1) Load your trained model and individual label‐encoders
model     = joblib.load('car_price_model.pkl')
le_brand  = joblib.load('le_brand.pkl')
le_model  = joblib.load('le_model.pkl')
le_fuel   = joblib.load('le_fuel.pkl')
le_seller = joblib.load('le_seller.pkl')
le_trans  = joblib.load('le_trans.pkl')
le_owner  = joblib.load('le_owner.pkl')
features  = joblib.load('features.pkl')   # list like ['year','km_driven',...,'brand','model']

# 2) Load your CSV and derive brand & model columns for dropdowns
df = pd.read_csv('Cardetails.csv')
df[['brand','model']] = df['name'].str.split(n=1, expand=True)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        u = request.form['username']; p = request.form['password']
        if u in users and check_password_hash(users[u], p):
            session['user'] = u
            return redirect(url_for('home'))
        error = "Invalid credentials"
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET','POST'])
def register():
    error = None
    if request.method == 'POST':
        u = request.form['username']; p = request.form['password']
        if u in users:
            error = "Username taken"
        else:
            users[u] = generate_password_hash(p)
            return redirect(url_for('login'))
    return render_template('register.html', error=error)

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session['user'])

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    # Build brand list for dropdown
    brands = sorted(df['brand'].unique())
    # Pass raw brand‐model pairs for client‐side filtering
    bm_data = df[['brand','model']].to_dict(orient='records')

    if request.method == 'POST':
        # 3) Read form inputs
        vals = {
            'year':        int(request.form['year']),
            'km_driven':   int(request.form['km_driven']),
            'mileage':     float(request.form['mileage']),
            'engine':      float(request.form['engine']),
            'max_power':   float(request.form['max_power']),
            'seats':       int(request.form['seats']),
            'brand':       request.form['brand'],
            'model':       request.form['model'],
            'fuel':        request.form['fuel'],
            'seller_type': request.form['seller_type'],
            'transmission':request.form['transmission'],
            'owner':       request.form['owner']
        }

        # 4) Encode categorical values
        enc = {
            'brand':       le_brand.transform([vals['brand']])[0],
            'model':       le_model.transform([vals['model']])[0],
            'fuel':        le_fuel.transform([vals['fuel']])[0],
            'seller_type': le_seller.transform([vals['seller_type']])[0],
            'transmission':le_trans.transform([vals['transmission']])[0],
            'owner':       le_owner.transform([vals['owner']])[0]
        }

        # 5) Build feature vector in the learned order
        x = [
            vals['year'],
            vals['km_driven'],
            enc['fuel'],
            enc['seller_type'],
            enc['transmission'],
            enc['owner'],
            vals['mileage'],
            vals['engine'],
            vals['max_power'],
            vals['seats'],
            enc['brand'],
            enc['model']
        ]
        X = np.array(x).reshape(1, -1)

        # 6) Predict
        price = model.predict(X)[0]
        return render_template('result.html', price=round(price,2))

    # GET: render the prediction form
    return render_template('predict.html', brands=brands, data=bm_data)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
