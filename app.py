from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
from flask_bcrypt import Bcrypt
import mysql.connector
import logging
from flask import Flask, render_template, request

import googlemaps
import folium
from folium.plugins import HeatMap
from flask import send_from_directory
import random

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = 'prasad'
bcrypt = Bcrypt(app)

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="2003",
        database="EPIDEMIC"
    )

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if not username or not email or not password:
            flash('All fields are required!', 'danger')
            return redirect(url_for('register'))
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                (username, email, password_hash)
            )
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            flash('Error: ' + str(err), 'danger')
        finally:
            cursor.close()
            conn.close()
    return render_template('auth.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user and bcrypt.check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['email'] = user['email']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('welcome'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('auth.html')

@app.route('/')
def home():
    return render_template('auth.html')


@app.route('/welcome')
def welcome():
    if 'user_id' in session:
        return render_template('home.html', username=session.get('username'))
    else:
        flash("Please log in to access this page.", "danger")
        return redirect(url_for('login'))

model = joblib.load('models/malaria_dengue_model.pkl')
gender_encoder = joblib.load('models/gender_encoder.pkl')
location_encoder = joblib.load('models/location_encoder.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

high_risk_cities = ['Bangalore', 'Delhi', 'Hyderabad', 'Kerala', 'Mumbai']

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = request.form['age']
    gender = request.form['gender']
    location = request.form['location']
    fever = int(request.form['fever'])
    headache = int(request.form['headache'])
    joint_pain = int(request.form['joint_pain'])
    muscle_pain = int(request.form['muscle_pain'])
    fatigue = int(request.form['fatigue'])
    nausea_vomiting = int(request.form['nausea_vomiting'])
    rash = int(request.form['rash'])
    chills = int(request.form['chills'])
    abdominal_pain = int(request.form['abdominal_pain'])
    bleeding = int(request.form['bleeding'])
    symptom_duration = int(request.form['symptom_duration'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    rainfall = float(request.form['rainfall'])

    gender_encoded = gender_encoder.transform([gender])[0]
    location_encoded = location_encoder.transform([location])[0]

    user_input = {
        'Age': age,
        'Gender': gender_encoded,
        'Location': location_encoded,
        'Fever': fever,
        'Headache': headache,
        'Joint Pain': joint_pain,
        'Muscle Pain': muscle_pain,
        'Fatigue': fatigue,
        'Nausea/Vomiting': nausea_vomiting,
        'Rash': rash,
        'Chills': chills,
        'Abdominal Pain': abdominal_pain,
        'Bleeding': bleeding,
        'Symptom Duration (Days)': symptom_duration,
        'Temperature (°C)': temperature,
        'Humidity (%)': humidity,
        'Rainfall (mm)': rainfall
    }

    disease = "Neither"
    if fever == 3 and headache == 3 and chills == 3:
        disease = "Malaria"
        probability = 95.0
    elif (headache >= 2 and joint_pain >= 2 and muscle_pain >= 2) or \
         (nausea_vomiting >= 2 and rash >= 2):
        disease = "Dengue"
        probability = 90.0
    else:

        user_input_df = pd.DataFrame([user_input])
        prediction = model.predict(user_input_df)
        probability = model.predict_proba(user_input_df).max(axis=1)[0] * 100



    high_risk_city = location in high_risk_cities

    severe_symptoms = [
        symptom for symptom, value in [
            ('Fever', fever), ('Headache', headache), ('Joint Pain', joint_pain),
            ('Muscle Pain', muscle_pain), ('Fatigue', fatigue), ('Chills', chills)
        ] if value == 3
    ]


    temperature_risk = "High temperature, conducive for mosquito activity." if 25 <= temperature <= 35 else "Low temperature, less favorable for mosquito activity."
    humidity_risk = "High humidity, favorable for mosquito breeding." if humidity > 80 else "Moderate or Low humidity, less favorable for breeding."
    rainfall_risk = "High rainfall, indicating mosquito breeding." if rainfall > 50 else "Low to moderate rainfall, less likely for breeding."


    return render_template(
        'result.html',
        disease=disease,
        probability=f"{probability:.2f}",
        city=location,
        high_risk_city=high_risk_city,
        severe_symptoms=', '.join(severe_symptoms) if severe_symptoms else 'None',
        temperature_risk=temperature_risk,
        humidity_risk=humidity_risk,
        rainfall_risk=rainfall_risk
    )


kmeans = joblib.load('models/kmeans_model.pkl')
scaler = joblib.load('models/scaler_model.pkl')
malaria_model = joblib.load('models/malaria_model.pkl')
dengue_model = joblib.load('models/dengue_model.pkl')

gmaps = googlemaps.Client(key='AIzaSyBDBEHZReEr8Zyc_MKNucPPSUkjMl6YhBA')

data = pd.read_csv('data/high_low_region.csv')



@app.route('/risk_region')
def risk_region():
    return render_template('risk_region.html')


@app.route('/detect', methods=['POST'])
def detect():
    location_name = request.form['location']
    time_choice = request.form['time_choice']
    value = int(request.form['value'])

    try:
        location = gmaps.geocode(location_name)
        if not location:
            return render_template('risk_region_result.html', error="Location not found.")
        latitude, longitude = location[0]['geometry']['location']['lat'], location[0]['geometry']['location']['lng']
    except Exception as e:
        return render_template('risk_region_result.html', error=f"Geocoder error: {e}")

    location_data = data[data['Location'] == location_name]
    if location_data.empty:
        return render_template('risk_region_result.html', error="Location data not found.")

    avg_features = location_data[['Temperature', 'Humidity', 'Rainfall', 'Malaria Cases',
                                  'Dengue Cases', 'Population Density', 'Water Body Nearby',
                                  'Green Cover', 'Healthcare Facilities']].mean()
    cluster = kmeans.predict(scaler.transform([avg_features]))[0]
    risk_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_level = risk_labels[cluster]

    if time_choice == "month":
        future = pd.DataFrame({'ds': pd.date_range(start=f"2024-{value:02}-01", periods=30, freq='D')})
    elif time_choice == "season":
        start_month = (value - 1) * 3 + 1
        future = pd.DataFrame({'ds': pd.date_range(start=f"2024-{start_month:02}-01", periods=90, freq='D')})

    malaria_forecast = malaria_model.predict(future)
    dengue_forecast = dengue_model.predict(future)

    avg_malaria_cases = malaria_forecast['yhat'].mean()
    avg_dengue_cases = dengue_forecast['yhat'].mean()
    probability_score = min(1.0, (avg_malaria_cases + avg_dengue_cases) / 100)

    predictions = [
        [latitude, longitude, probability_score]
    ]
    generate_heatmap(predictions)

    return render_template(
        'risk_region_result.html',
        location=location_name,
        risk_level=risk_level,
        probability_score=f"{probability_score:.2f}",
        factors="High Disease Cases, Environmental Conditions"
    )

other_cities = [
    {"name": "Surat", "lat": 21.1702, "lon": 72.8311},
    {"name": "Vadodara", "lat": 22.3075, "lon": 73.1812},
    {"name": "Indore", "lat": 22.7196, "lon": 75.8577},
    {"name": "Nagpur", "lat": 21.1458, "lon": 79.0882},
    {"name": "Pune", "lat": 18.5204, "lon": 73.8567},
    {"name": "Bhubaneswar", "lat": 20.2961, "lon": 85.8189},
    {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462},
    {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714},
    {"name": "Hyderabad", "lat": 17.3854, "lon": 78.4867},
    {"name": "Guwahati", "lat": 26.1445, "lon": 91.7362},
    {"name": "Coimbatore", "lat": 11.0168, "lon": 76.9558},
    {"name": "Visakhapatnam", "lat": 17.6869, "lon": 83.2185},
    {"name": "Madurai", "lat": 9.9250, "lon": 78.1193},
    {"name": "Raipur", "lat": 21.2514, "lon": 81.6296},
    {"name": "Srinagar", "lat": 34.0836, "lon": 74.7973},
]

@app.route('/heatmap')
def heatmap():
    return send_from_directory('output', 'output.html')


def generate_heatmap(predictions):
    mapObj = folium.Map(location=[23.294059708387206, 78.26660156250001], zoom_start=6)

    bordersStyle = {
        'color': 'green',
        'weight': 1,
        'fillOpacity': 0.1,
    }
    folium.GeoJson('data/states_india.geojson', name='India', style_function=lambda x: bordersStyle).add_to(mapObj)
    folium.GeoJson('data/srilanka.geojson', name='Sri Lanka', style_function=lambda x: bordersStyle).add_to(mapObj)

    heatmap_data = []
    for prediction in predictions:
        lat, lon, risk_score = prediction
        heatmap_data.append([lat, lon, risk_score])

    heatmap_layer = HeatMap(heatmap_data, radius=20, blur=15, min_opacity=0.5)
    heatmap_layer.add_to(mapObj)

    circle_layer = folium.FeatureGroup(name="Predictions Circles")
    for prediction in predictions:
        lat, lon, risk_score = prediction
        color = 'red' if risk_score >= 0.7 else 'yellow' if risk_score >= 0.4 else 'blue'

        folium.Circle(location=[lat, lon],
                      radius=50000,
                      weight=5,
                      color=color,
                      fill_color=color,
                      fill_opacity=0.6,
                      tooltip=f"Risk Level: {risk_score:.2f}",
                      popup=folium.Popup(
                          f"<h2>Risk Prediction: {risk_score:.2f}</h2><p>Latitude: {lat}, Longitude: {lon}</p>",
                          max_width=500)
                      ).add_to(circle_layer)

    circle_layer.add_to(mapObj)
    # Other cities with random probability scores
    other_cities_layer = folium.FeatureGroup(name="Other Cities Circles", show=False)
    for city in other_cities:
        lat = city["lat"]
        lon = city["lon"]
        # Assign a random risk score for other cities
        risk_score = round(random.uniform(0.0, 1.0), 2)
        color = 'red' if risk_score >= 0.7 else 'yellow' if risk_score >= 0.4 else 'blue'

        folium.Circle(location=[lat, lon],
                      radius=50000,
                      weight=5,
                      color=color,
                      fill_color=color,
                      fill_opacity=0.6,
                      tooltip=f"Risk Level: {risk_score:.2f}",
                      popup=folium.Popup(
                          f"<h2>Risk Prediction: {risk_score:.2f}</h2><p>Latitude: {lat}, Longitude: {lon}</p>",
                          max_width=500)
                      ).add_to(other_cities_layer)

    other_cities_layer.add_to(mapObj)

    folium.LayerControl().add_to(mapObj)

    mapObj.save('output/output.html')




@app.route('/logout')
def logout():
   return redirect("/")

if __name__ == '__main__':
    app.run(debug=True)
