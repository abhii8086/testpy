import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import numpy as np
from flask import Flask, jsonify, render_template, send_file
from flask_jwt_extended import JWTManager
from config import Config
from controllers.user_controller import user_bp
from flask_cors import CORS
import pandas as pd
from sklearn.cluster import KMeans
import folium
import seaborn as sns
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)
jwt = JWTManager(app)

def convert_nan_to_none(value):
    """ Convert NaN values to None for JSON serialization """
    if pd.isna(value):
        return None
    return value

@app.route('/clusters', methods=['GET'])
def get_clusters():
    # Load the CSV file
    df = pd.read_csv('csv/Mumbai_NaviMumbai.csv')
    
    # Clean up data (assuming these columns exist in the CSV)
    df.drop(['Phone', 'Phones', 'Claimed', 'Review URL', 'Website', 'Domain', 
             'Cid', 'Place Id', 'Kgmid', 'Plus code', 'Google Knowledge URL', 
             'Email', 'Social Medias', 'Facebook', 'Instagram', 'Twitter', 'Yelp'], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains('^Unnamed')], axis=1, inplace=True)
    df.dropna(inplace=True)
    
    # Make cluster on the basis of Latitude and Longitude
    X = df[['Latitude', 'Longitude']]
    
    # Define the number of clusters
    kmeans = KMeans(n_clusters=11, random_state=0)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Convert NaN values to None
    df = df.apply(lambda x: x.map(convert_nan_to_none))
    
    # Define a color map for different amenities
    entity_colors = {
        'College': 'red',
        'Garden': 'green',
        'Gym': 'blue',
        'Hospital': 'orange',
        'Hotel': 'purple',
        'Market': 'gray',
        'Pharmacy': 'pink',
        'Restaurant': 'yellow',
        'School': 'white',
        'Supermarket': 'magenta',
        'Touristattraction': 'white'
    }

    # Create a map with no specific location (we will set bounds later)
    m = folium.Map(zoom_start=10)

    # List to store all coordinates for setting the map bounds later
    locations = []

    # Add points to the map with custom icons and colors
    for idx, row in df.iterrows():
        amenity = row['Categories']
        color = entity_colors.get(amenity, 'gray')  # Default to gray if not found in entity_colors
        location = [row['Latitude'], row['Longitude']]
        locations.append(location)  # Collect coordinates for setting bounds
        
        folium.CircleMarker(
            location=location,
            radius=8,  # Adjust the size of the dot as needed
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"{amenity} (Cluster {row['cluster']})"
        ).add_to(m)

    # Automatically adjust the zoom to fit all markers (use the bounds of all locations)
    m.fit_bounds(locations)

    # Save the map to the templates directory
    m.save('templates/map.html')

    return jsonify(df.to_dict(orient='records'))

@app.route('/map')
def render_map():
    return render_template('map.html')

@app.route('/scatterplot')
def get_scatterplot():
    # Load the CSV data (update the path according to your local setup)
    df = pd.read_csv('csv/Mumbai_NaviMumbai.csv')  # Ensure that 'navimumbaiData.csv' is in the 'static' directory
    
    # Select Latitude and Longitude for clustering
    X = df[['Latitude', 'Longitude']]
    
    # Define and fit KMeans with 11 clusters
    kmeans = KMeans(n_clusters=11, random_state=0)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Create the scatter plot
    plt.figure(figsize=(5, 6))
    sns.scatterplot(data=df, x='Longitude', y='Latitude', hue='cluster', palette='tab10')
    plt.title('K-means Clustering of Mumbai Entities')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Ensure the 'static' directory exists
    static_dir = os.path.join(os.getcwd(), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    # Save the plot to the 'static' folder
    plot_path = os.path.join(static_dir, 'scatter_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Return the plot image
    return send_file(plot_path, mimetype='image/png')

# Register Blueprints
app.register_blueprint(user_bp)
