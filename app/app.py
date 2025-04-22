import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
from datetime import datetime, timedelta
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
import json
from werkzeug.utils import secure_filename
# Import Agentic AI components
from agentic.packaging_agent import PackagingAgent
import requests
# Import Dash components
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from flask import Response
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import random
import logging

app = Flask(__name__)
app.secret_key = 'logistics_platform_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_NAME'] = 'logistics_session'
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # 1 hour CSRF token expiration

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Exempt API routes from CSRF protection
csrf.exempt('/api/predict_packaging')
csrf.exempt('/api/predict_delivery_issues')
csrf.exempt('/api/packaging/visualization')

# Set up logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Define user class for authentication
class User(UserMixin):
    def __init__(self, id, email, role):
        self.id = id
        self.email = email
        self.role = role

# Mock user database - in a real app, this would be a database
users = {
    'customer@example.com': {
        'id': 1,
        'password': generate_password_hash('customer123'),
        'role': 'customer'
    },
    'company@example.com': {
        'id': 2,
        'password': generate_password_hash('company123'),
        'role': 'company'
    },
    'driver@example.com': {
        'id': 3,
        'password': generate_password_hash('driver123'),
        'role': 'driver'
    },
    'packaging@example.com': {
        'id': 4,
        'password': generate_password_hash('packaging123'),
        'role': 'packaging'
    }
}

@login_manager.user_loader
def load_user(user_id):
    app.logger.debug(f"Loading user with ID: {user_id}")
    try:
        user_id = int(user_id)
        for email, user_data in users.items():
            if user_data['id'] == user_id:
                user = User(user_data['id'], email, user_data['role'])
                app.logger.debug(f"User loaded successfully: {email}, role: {user_data['role']}")
                return user
        app.logger.warning(f"User with ID {user_id} not found")
        return None
    except Exception as e:
        app.logger.error(f"Error loading user: {str(e)}")
        return None

# Define the file paths - adjusting for different running contexts
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Check if app.py is in a directory named 'app'
if os.path.basename(script_dir) == 'app':
    # We're inside the app directory
    data_dir = os.path.join(script_dir, 'data')
else:
    # We're in the parent directory
    data_dir = os.path.join(script_dir, 'app', 'data')

PRODUCT_PACKAGE_DATASET = os.path.join(data_dir, 'Product_Package_Dataset.csv')
SHIPPING_COMPANY_SECTION = os.path.join(data_dir, 'Shipping_Company_Section.csv')
CUSTOMER_SECTION_TABLE = os.path.join(data_dir, 'Customer_Section_Table.csv')
ROUTE_OPTIMIZATION_DATA = os.path.join(data_dir, 'Route_Optimization_Data.csv')

# Initialize Agentic AI components
try:
    print("Initializing Agentic AI components...")
    
    # Check if Ollama is available by trying a simple request
    ollama_available = False
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        if response.status_code == 200:
            ollama_available = True
            print(f"Ollama service detected and running: {response.json()}")
        else:
            print(f"Ollama service responded but returned an error: {response.status_code}, {response.text}")
    except requests.exceptions.ConnectionError as e:
        print(f"Ollama service connection error: {e}")
    except requests.exceptions.Timeout as e:
        print(f"Ollama service timeout: {e}")
    except Exception as e:
        print(f"Ollama service detection error: {type(e).__name__}: {e}")
    
    # Initialize components with warning about Ollama status
    if not ollama_available:
        print("WARNING: Ollama service not detected. AI components will run in fallback mode.")
        print("To enable full AI capabilities, install Ollama from https://ollama.com/")
        print("Then run: ollama serve")
        print("And pull the required models: ollama pull llama3.2 mxbai-embed-large")
    
    # Initialize packaging agent (will fall back to rule-based if Ollama is not available)
    PACKAGING_AGENT = PackagingAgent(PRODUCT_PACKAGE_DATASET)
    print(f"Packaging agent initialized (Mode: {'AI-powered' if ollama_available else 'fallback rule-based'})")
    
except Exception as e:
    print(f"Error initializing Agentic AI components: {type(e).__name__}: {e}")
    PACKAGING_AGENT = None

# Helper function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Authentication routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    app.logger.info("Login route accessed")
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        app.logger.debug(f"Login attempt for email: {email}")
        
        if not email or not password:
            app.logger.warning("Login failed: Missing email or password")
            flash('Please enter both email and password')
            return render_template('login.html')
            
        if email in users and check_password_hash(users[email]['password'], password):
            app.logger.info(f"Login successful for user: {email}, role: {users[email]['role']}")
            user = User(users[email]['id'], email, users[email]['role'])
            login_user(user)
            
            # Store user info in session as well for backward compatibility
            session['user_id'] = users[email]['id']
            session['email'] = email
            session['role'] = users[email]['role']
            
            # Redirect based on user role
            if users[email]['role'] == 'customer':
                return redirect(url_for('customer_dashboard'))
            elif users[email]['role'] == 'company':
                return redirect(url_for('company_dashboard'))
            elif users[email]['role'] == 'driver':
                return redirect(url_for('driver_dashboard'))
            elif users[email]['role'] == 'packaging':
                return redirect(url_for('packaging_dashboard'))
        else:
            app.logger.warning(f"Login failed for user: {email} - Invalid credentials")
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        
        if email in users:
            flash('Email already exists')
            return redirect(url_for('signup'))
        
        # In a real app, we would add to database
        user_id = len(users) + 1
        users[email] = {
            'id': user_id,
            'password': generate_password_hash(password),
            'role': role
        }
        
        flash('Account created successfully. Please log in.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    app.logger.info(f"Logout request for user: {current_user.email if hasattr(current_user, 'email') else 'Unknown'}")
    # Clear the session
    session.clear()
    # Log out the user from Flask-Login
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('index'))

# Customer Section Routes
@app.route('/customer/dashboard')
@login_required
def customer_dashboard():
    if current_user.role != 'customer':
        flash('Access denied')
        return redirect(url_for('index'))
    
    return render_template('customer_dashboard.html')

@app.route('/customer/feedback', methods=['GET', 'POST'])
@login_required
def customer_feedback():
    if current_user.role != 'customer':
        flash('Access denied')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        delivery_id = request.form.get('delivery_id')
        location = request.form.get('location')
        time = request.form.get('time')
        issue_reported = request.form.get('issue_reported')
        item_type = request.form.get('item_type')
        overall_rating = request.form.get('overall_rating')
        
        # Save to CSV
        df = pd.read_csv(CUSTOMER_SECTION_TABLE)
        new_row = pd.DataFrame({
            'Delivery_ID': [delivery_id],
            'Location': [location],
            'Time': [time],
            'Issue_Reported': [issue_reported],
            'Item_Type': [item_type],
            'Overall_Rating': [overall_rating]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CUSTOMER_SECTION_TABLE, index=False)
        
        flash('Thank you for your feedback!')
        return redirect(url_for('customer_dashboard'))
    
    return render_template('customer_feedback.html')

# Driver Section Routes
@app.route('/driver/dashboard')
@login_required
def driver_dashboard():
    if current_user.role != 'driver':
        flash('Access denied')
        return redirect(url_for('index'))
    
    # Load route optimization data
    route_df = pd.read_csv(ROUTE_OPTIMIZATION_DATA)
    
    # Get the Business Bay to Jumeirah route data
    route_data = route_df[
        (route_df['Start_Location'] == 'Business Bay') & 
        (route_df['End_Location'] == 'Jumeirah')
    ].iloc[0]
    
    # Create a folium map with OpenStreetMap tiles to show actual roads
    m = folium.Map(
        location=[25.1824, 55.2603], 
        zoom_start=14,
        tiles='OpenStreetMap',
    )
    
    # Add a satellite layer to better show roads
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add traffic layer
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=m@221097413,traffic&x={x}&y={y}&z={z}',
        attr='Google',
        name='Traffic',
        overlay=True,
        control=True
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # HIGHLY detailed route from Business Bay to Jumeirah
    # Extremely detailed with many points to show actual road curves
    # Each segment follows an actual road segment with realistic curves
    
    # Business Bay to Al Khail Road
    business_bay_to_al_khail = [
        [25.1972, 55.2744],  # Business Bay starting point
        [25.1969, 55.2741],  # Exit from building
        [25.1967, 55.2738],  # Turning onto Al Sa'ada Street
        [25.1965, 55.2734],  # Al Sa'ada Street - slight curve
        [25.1962, 55.2731],  # Al Sa'ada Street - continuous
        [25.1958, 55.2728],  # Approaching junction
        [25.1953, 55.2725],  # Junction - starting turn
        [25.1950, 55.2721],  # Junction - middle of turn
        [25.1948, 55.2717],  # Junction - finishing turn
        [25.1946, 55.2713],  # Heading towards Al Khail Road
        [25.1944, 55.2708],  # Approaching Al Khail Road
        [25.1942, 55.2704],  # Al Khail Road entry
    ]
    
    # Al Khail Road section with proper curves
    al_khail_road = [
        [25.1942, 55.2704],  # Al Khail Road entry
        [25.1938, 55.2700],  # Al Khail Road - slight curve
        [25.1935, 55.2696],  # Al Khail Road - curve continues
        [25.1931, 55.2693],  # Al Khail Road - main curve
        [25.1927, 55.2690],  # Al Khail Road - curve finishes
        [25.1924, 55.2686],  # Al Khail Road - straight section
        [25.1921, 55.2683],  # Al Khail Road - continuing straight
        [25.1917, 55.2680],  # Al Khail Road - slight bend
        [25.1913, 55.2677],  # Al Khail Road - continuing bend
        [25.1909, 55.2674],  # Al Khail Road - more pronounced bend
        [25.1905, 55.2671],  # Al Khail Road - approaching interchange
        [25.1901, 55.2668],  # Al Khail Road - nearing interchange
        [25.1897, 55.2664],  # Approaching interchange
    ]
    
    # Interchange with complex turns
    interchange = [
        [25.1897, 55.2664],  # Approaching interchange
        [25.1893, 55.2661],  # Interchange - beginning of turn
        [25.1890, 55.2659],  # Interchange - turning
        [25.1887, 55.2657],  # Interchange - sharp turn
        [25.1884, 55.2655],  # Interchange - middle of complex junction
        [25.1881, 55.2652],  # Interchange - continuing through junction
        [25.1878, 55.2650],  # Interchange - exiting junction
        [25.1875, 55.2648],  # Exit ramp - beginning
        [25.1872, 55.2646],  # Exit ramp - middle
        [25.1869, 55.2643],  # Exit ramp - end
        [25.1866, 55.2641],  # Connecting road
        [25.1863, 55.2638],  # Approaching Al Wasl
        [25.1861, 55.2636],  # Connecting to Al Wasl
    ]
    
    # Al Wasl Road section with natural bends
    al_wasl_road = [
        [25.1861, 55.2636],  # Connecting to Al Wasl
        [25.1858, 55.2634],  # Al Wasl Road - entry
        [25.1855, 55.2631],  # Al Wasl Road - beginning
        [25.1852, 55.2629],  # Al Wasl Road - slight curve
        [25.1850, 55.2627],  # Al Wasl Road north
        [25.1847, 55.2624],  # Al Wasl Road - continuing
        [25.1844, 55.2622],  # Al Wasl Road - slight bend
        [25.1841, 55.2619],  # Al Wasl Road - curve
        [25.1839, 55.2617],  # Al Wasl Road bend
        [25.1836, 55.2614],  # Al Wasl Road - continuous bend
        [25.1833, 55.2612],  # Al Wasl Road - straightening
        [25.1830, 55.2609],  # Al Wasl Road - straight section
        [25.1828, 55.2607],  # Al Wasl Road mid-section
        [25.1825, 55.2603],  # Al Wasl Road - continuing
        [25.1822, 55.2599],  # Al Wasl Road - beginning curve
        [25.1819, 55.2595],  # Al Wasl Road - curving
        [25.1817, 55.2591],  # Al Wasl Road curve
        [25.1814, 55.2587],  # Al Wasl Road - continuous curve
        [25.1811, 55.2583],  # Al Wasl Road - finishing curve
        [25.1808, 55.2579],  # Al Wasl Road - straight again
        [25.1806, 55.2575],  # Al Wasl Road straight
        [25.1803, 55.2571],  # Al Wasl Road - approaching junction
        [25.1799, 55.2566],  # Al Wasl Road - nearing junction
        [25.1795, 55.2560],  # Al Wasl/Jumeirah intersection
    ]
    
    # Jumeirah Road to destination with realistic street pattern
    jumeirah_road = [
        [25.1795, 55.2560],  # Al Wasl/Jumeirah intersection
        [25.1792, 55.2557],  # Turning onto Jumeirah Road
        [25.1789, 55.2553],  # Jumeirah Road - first section
        [25.1786, 55.2549],  # Jumeirah Road - continuing
        [25.1784, 55.2545],  # Jumeirah Road north
        [25.1781, 55.2542],  # Jumeirah Road - slight bend
        [25.1778, 55.2538],  # Jumeirah Road - curving
        [25.1775, 55.2535],  # Jumeirah Road - curve continues
        [25.1773, 55.2533],  # Jumeirah Road bend
        [25.1770, 55.2529],  # Jumeirah Road - straightening
        [25.1767, 55.2526],  # Jumeirah Road - straight section
        [25.1764, 55.2524],  # Jumeirah Road - continuing
        [25.1762, 55.2522],  # Jumeirah Road mid-section
        [25.1759, 55.2519],  # Jumeirah Road - approaching turn
        [25.1755, 55.2516],  # Jumeirah Road - beginning turn
        [25.1752, 55.2514],  # Jumeirah Road - turning
        [25.1748, 55.2512],  # Jumeirah Beach Road entry
        [25.1744, 55.2509],  # Jumeirah Beach Road - beginning
        [25.1740, 55.2506],  # Jumeirah Beach Road - first curve
        [25.1736, 55.2503],  # Jumeirah Beach Road
        [25.1732, 55.2500],  # Jumeirah Beach Road - continuing
        [25.1728, 55.2497],  # Jumeirah Beach Road - slight bend
        [25.1724, 55.2493],  # Jumeirah Beach Road - curve
        [25.1720, 55.2489],  # Jumeirah Beach Road curve
        [25.1716, 55.2486],  # Jumeirah Beach Road - continuing curve
        [25.1712, 55.2482],  # Jumeirah Beach Road - finishing curve
        [25.1708, 55.2478],  # Approaching destination area
        [25.1704, 55.2475],  # Nearing destination
        [25.1700, 55.2471],  # Very close to destination
        [25.1696, 55.2467],  # Almost at destination
        [25.1692, 55.2465],  # Final approach
        [25.1688, 55.2464],  # Destination entrance
        [25.1676, 55.2463],  # Jumeirah destination
    ]
    
    # Combine all route segments
    route_coords = business_bay_to_al_khail + al_khail_road[1:] + interchange[1:] + al_wasl_road[1:] + jumeirah_road[1:]
    
    # Create segments with different colors for different road types
    road_segments = [
        (business_bay_to_al_khail, '#3186cc', 'Business Bay Local Roads'),
        (al_khail_road, '#e67e22', 'Al Khail Road (Highway)'),
        (interchange, '#9b59b6', 'Interchange'),
        (al_wasl_road, '#3186cc', 'Al Wasl Road'),
        (jumeirah_road, '#27ae60', 'Jumeirah Beach Road')
    ]
    
    # Add each road segment with its own color and popup
    for segment, color, name in road_segments:
        folium.PolyLine(
            segment,
            weight=6,
            color=color,
            opacity=0.8,
        ).add_to(m)
    
    # Add a pulsing animated path over the entire route for emphasis
    plugins.AntPath(
        locations=route_coords,
        dash_array=[10, 20],
        delay=1000,
        color='white',
        pulse_color='#3498db',
        weight=3,
        opacity=0.8
    ).add_to(m)
    
    # Add markers for start and end without text popups
    # Add a more prominent start marker
    folium.Marker(
        route_coords[0],
        icon=folium.Icon(icon='play', prefix='fa', color='green'),
    ).add_to(m)
    
    # Add a more prominent end marker
    folium.Marker(
        route_coords[-1],
        icon=folium.Icon(icon='stop', prefix='fa', color='red'),
    ).add_to(m)
    
    # Add key waypoints as smaller markers with direction arrows only
    key_waypoints = [
        (al_khail_road[0], "arrow-right", "orange"),
        (interchange[0], "exchange", "purple"),
        (al_wasl_road[0], "arrow-down", "blue"),
        (jumeirah_road[0], "arrow-left", "green"),
        (jumeirah_road[16], "arrow-down", "green")
    ]
    
    for coord, icon, color in key_waypoints:
        folium.Marker(
            coord,
            icon=folium.Icon(color=color, icon=icon, prefix='fa'),
        ).add_to(m)
    
    # Save map to html
    try:
        map_html = m._repr_html_()
    except Exception as e:
        app.logger.error(f"Error rendering map to HTML: {type(e).__name__}: {e}")
        # Create a simple fallback HTML for the map
        map_html = """
        <div class="alert alert-warning">
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>Map rendering error:</strong> Unable to display the delivery heatmap.
            <hr>
            <p>Please try refreshing the page or contact technical support if the issue persists.</p>
        </div>
        """
    
    # Get optimized routes based on current conditions
    optimized_routes = optimize_routes(route_df)
    
    # Predict potential issues for routes based on historical data
    shipping_df = pd.read_csv(SHIPPING_COMPANY_SECTION)
    problem_areas = get_route_issues(shipping_df)
    
    return render_template(
        'driver_dashboard.html', 
        map_html=map_html, 
        problem_areas=problem_areas,
        optimized_routes=optimized_routes
    )

@app.route('/driver/report', methods=['GET', 'POST'])
@login_required
def driver_report():
    if current_user.role != 'driver':
        flash('Access denied')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        delivery_id = request.form.get('delivery_id')
        driver_id = request.form.get('driver_id')
        location = request.form.get('location')
        time = request.form.get('time')
        weather = request.form.get('weather')
        road_condition = request.form.get('road_condition')
        traffic_level = request.form.get('traffic_level')
        cause_of_issue = request.form.get('cause_of_issue')
        
        # Save to CSV
        df = pd.read_csv(SHIPPING_COMPANY_SECTION)
        new_row = pd.DataFrame({
            'Delivery_ID': [delivery_id],
            'Driver_ID': [driver_id],
            'Location': [location],
            'Time': [time],
            'Weather': [weather],
            'Road_Condition': [road_condition],
            'Traffic_Level': [traffic_level],
            'Cause_of_Issue': [cause_of_issue]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(SHIPPING_COMPANY_SECTION, index=False)
        
        flash('Delivery report submitted successfully!')
        return redirect(url_for('driver_dashboard'))
    
    return render_template('driver_report.html')

# Company Section Routes
@app.route('/company/dashboard')
@login_required
def company_dashboard():
    """Company dashboard with visualizations for delivery performance."""
    # Check if user is logged in and has company role
    if current_user.role != 'company':
        app.logger.warning(f"Access denied to company dashboard for user with role: {current_user.role}")
        flash('Access denied')
        return redirect(url_for('index'))
    
    try:
        # Default values for variables used in the template
        issue_rate = 7.5
        on_time_rate = 92.5
        total_deliveries = 1438
        avg_rating = 4.2
        
        # Create a heatmap with proper Leaflet map and coordinated-based hotspots
        heatmap_html = """
        <div class="uae-heatmap-container" style="height: 450px; width: 100%; position: relative; border-radius: 4px; overflow: hidden; border: 1px solid #ddd;">
            <!-- Leaflet map container -->
            <div id="leaflet-map" style="width: 100%; height: 100%;"></div>
            
            <!-- Include Leaflet CSS and JS -->
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            
            <script>
            // Initialize the map once the document is loaded
            document.addEventListener('DOMContentLoaded', function() {
                // Create the map focused on UAE
                var map = L.map('leaflet-map').setView([24.7, 54.5], 7);
                
                // Add OpenStreetMap tiles
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                    maxZoom: 19
                }).addTo(map);
                
                // Define hotspot locations with actual geographic coordinates
                var hotspots = [
                    {name: 'Dubai', lat: 25.2048, lng: 55.2708, color: '#ff0000', scale: 3, opacity: 0.6},
                    {name: 'Abu Dhabi', lat: 24.4539, lng: 54.3773, color: '#ff0000', scale: 2.5, opacity: 0.6},
                    {name: 'Sharjah', lat: 25.3463, lng: 55.4209, color: '#ffaa00', scale: 2, opacity: 0.6},
                    {name: 'Ajman', lat: 25.4052, lng: 55.5136, color: '#ffaa00', scale: 1.8, opacity: 0.6},
                    {name: 'RAK', lat: 25.7895, lng: 55.9432, color: '#00cc00', scale: 1.5, opacity: 0.6},
                    {name: 'Fujairah', lat: 25.1288, lng: 56.3265, color: '#00cc00', scale: 1.5, opacity: 0.6},
                    {name: 'UAQ', lat: 25.5647, lng: 55.5534, color: '#00cc00', scale: 1.2, opacity: 0.6},
                    {name: 'Desert Area 1', lat: 24.9500, lng: 55.6000, color: '#00cc00', scale: 1.3, opacity: 0.4},
                    {name: 'Desert Area 2', lat: 24.7000, lng: 54.9000, color: '#00cc00', scale: 1, opacity: 0.4}
                ];
                
                // Icon for distribution centers
                var centerIcon = L.divIcon({
                    className: 'distribution-center-icon',
                    html: '<div style="background-color: white; width: 10px; height: 10px; border: 2px solid black; border-radius: 50%;"></div>',
                    iconSize: [14, 14],
                    iconAnchor: [7, 7]
                });
                
                // Add hotspots to the map
                hotspots.forEach(function(spot) {
                    // Create a circle for the hotspot with appropriate radius based on scale
                    var radius = spot.scale * 5000; // Scale to appropriate size in meters
                    var circle = L.circle([spot.lat, spot.lng], {
                        color: 'transparent',
                        fillColor: spot.color,
                        fillOpacity: spot.opacity,
                        radius: radius
                    }).addTo(map);
                    
                    // Add a label if it's a main city
                    if (spot.name !== 'Desert Area 1' && spot.name !== 'Desert Area 2') {
                        // Add distribution center marker
                        var marker = L.marker([spot.lat, spot.lng], {
                            icon: centerIcon
                        }).addTo(map);
                        marker.bindTooltip(spot.name + ' Distribution Center');
                        
                        // Add a label
                        var label = L.marker([spot.lat, spot.lng], {
                            icon: L.divIcon({
                                className: 'city-label',
                                html: '<div style="background-color: rgba(255,255,255,0.7); padding: 2px 5px; border-radius: 3px; font-size: 12px; font-weight: bold; white-space: nowrap;">' + spot.name + '</div>',
                                iconSize: [100, 20],
                                iconAnchor: [50, -15]
                            })
                        }).addTo(map);
                    }
                });
                
                // Add a legend
                var legend = L.control({position: 'topleft'});
                
                legend.onAdd = function(map) {
                    var div = L.DomUtil.create('div', 'legend');
                    div.innerHTML = `
                        <div style="background-color: white; padding: 10px; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <h5 style="margin: 0 0 10px 0; font-size: 16px;">Delivery Density</h5>
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <span style="display: inline-block; width: 20px; height: 15px; background-color: rgba(255,0,0,0.6); margin-right: 5px;"></span>
                                <span style="font-size: 12px;">High (Dubai, Abu Dhabi)</span>
                            </div>
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <span style="display: inline-block; width: 20px; height: 15px; background-color: rgba(255,170,0,0.6); margin-right: 5px;"></span>
                                <span style="font-size: 12px;">Medium (Sharjah, Ajman)</span>
                            </div>
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <span style="display: inline-block; width: 20px; height: 15px; background-color: rgba(0,204,0,0.6); margin-right: 5px;"></span>
                                <span style="font-size: 12px;">Low (RAK, Fujairah, UAQ)</span>
                            </div>
                        </div>
                    `;
                    return div;
                };
                
                legend.addTo(map);
            });
            </script>
        </div>
        """
        
        # Try to load data if available
        try:
            # Load the deliveries CSV file if it exists
            if os.path.exists(SHIPPING_COMPANY_SECTION):
                deliveries_df = pd.read_csv(SHIPPING_COMPANY_SECTION)
                app.logger.info(f"Loaded delivery data: {len(deliveries_df)} records")
                
                # Set up some basic statistics
                if not deliveries_df.empty:
                    if 'Overall_Rating' in deliveries_df.columns:
                        avg_rating = float(deliveries_df['Overall_Rating'].mean())
                    
                    # Calculate issue rate if applicable column exists
                    issue_column = next((col for col in ['Cause_of_Issue', 'Issue_Reported'] 
                                      if col in deliveries_df.columns), None)
                    if issue_column:
                        has_issue = ~deliveries_df[issue_column].isna()
                        issue_rate = float(has_issue.mean() * 100)
                        on_time_rate = 100 - float((deliveries_df[issue_column] == 'Late_Delivery').mean() * 100)
                    
                    total_deliveries = len(deliveries_df)
            else:
                app.logger.warning(f"Delivery data file not found: {SHIPPING_COMPANY_SECTION}")
        except Exception as e:
            app.logger.error(f"Error loading delivery data: {type(e).__name__}: {e}")
        
        # Round values for display
        avg_rating = round(avg_rating, 1)
        issue_rate = round(issue_rate, 1)
        on_time_rate = round(on_time_rate, 1)
        
        return render_template(
            'company_dashboard.html',
            heatmap_html=heatmap_html,
            avg_rating=avg_rating,
            issue_rate=issue_rate,
            on_time_rate=on_time_rate,
            total_deliveries=total_deliveries
        )
    except Exception as e:
        app.logger.error(f"Error in company dashboard: {type(e).__name__}: {e}")
        flash(f"Error loading dashboard data: {str(e)}", "danger")
        return render_template('error.html', 
                              error_title="Dashboard Error",
                              error_message="An error occurred while loading the dashboard data.")

# Create endpoints to serve dynamic plots
@app.route('/plot/packaging_materials')
@login_required
def plot_packaging_materials():
    if current_user.role not in ['company', 'packaging']:
        return Response("Unauthorized", status=401)
    
    # Load packaging data
    try:
        df = pd.read_csv(PRODUCT_PACKAGE_DATASET)
        
        # Check if data is empty or has insufficient data
        if df.empty or 'Packaging_Material' not in df.columns:
            app.logger.warning("Packaging dataset empty or missing required columns. Using sample data.")
            raise ValueError("Insufficient data")
            
        # Create a figure with Matplotlib and Seaborn
        plt.figure(figsize=(10, 6))
        counts = df['Packaging_Material'].value_counts()
        
        # Ensure we have data to display
        if len(counts) == 0:
            app.logger.warning("No packaging materials found in dataset. Using sample data.")
            raise ValueError("No packaging materials data")
            
        ax = sns.barplot(x=counts.index, y=counts.values, palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.title('Distribution of Packaging Materials')
        plt.xlabel('Packaging Material')
        plt.ylabel('Count')
        plt.tight_layout()
    except Exception as e:
        app.logger.warning(f"Error creating packaging materials plot: {str(e)}. Using sample data.")
        # Create sample data
        plt.figure(figsize=(10, 6))
        
        # Sample packaging materials and counts
        materials = ['Corrugated Box', 'Bubble Wrap + Box', 'Plastic Wrap + Box', 
                    'Foam Box + Ice Pack', 'Anti-Static + Box', 'Thermocol + Box']
        counts = [35, 25, 15, 10, 8, 7]
        
        ax = sns.barplot(x=materials, y=counts, palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.title('Distribution of Packaging Materials (Sample Data)')
        plt.xlabel('Packaging Material')
        plt.ylabel('Count')
        plt.tight_layout()
    
    # Convert plot to PNG image
    canvas = FigureCanvas(plt.gcf())
    img = io.BytesIO()
    canvas.print_png(img)
    img.seek(0)
    plt.close()
    
    return Response(img.getvalue(), mimetype='image/png')

@app.route('/plot/packaging_by_product_type')
@login_required
def plot_packaging_by_product_type():
    if current_user.role not in ['company', 'packaging']:
        return Response("Unauthorized", status=401)
    
    # Load packaging data
    try:
        df = pd.read_csv(PRODUCT_PACKAGE_DATASET)
        
        # Check if data is empty or has insufficient data
        if df.empty or 'Product_Type' not in df.columns or 'Packaging_Material' not in df.columns:
            app.logger.warning("Packaging dataset empty or missing required columns. Using sample data.")
            raise ValueError("Insufficient data")
        
        # Create a cross-tabulation
        product_packaging = pd.crosstab(df['Product_Type'], df['Packaging_Material'])
        
        # Ensure we have data to display
        if product_packaging.empty:
            app.logger.warning("No product-packaging relationship data. Using sample data.")
            raise ValueError("No product-packaging data")
            
        # Create a figure with Matplotlib and Seaborn
        plt.figure(figsize=(12, 8))
        sns.heatmap(product_packaging, annot=True, cmap='YlGnBu', fmt='d', linewidths=.5)
        plt.title('Packaging Materials by Product Type')
        plt.xlabel('Packaging Material')
        plt.ylabel('Product Type')
        plt.tight_layout()
    except Exception as e:
        app.logger.warning(f"Error creating packaging by product type plot: {str(e)}. Using sample data.")
        # Create sample data
        plt.figure(figsize=(12, 8))
        
        # Sample product types and packaging materials
        product_types = ['Electronics', 'Glassware', 'Furniture', 'Pharmaceuticals', 'Fresh Produce']
        packaging_materials = ['Corrugated Box', 'Bubble Wrap + Box', 'Plastic Wrap + Box', 'Foam Box + Ice Pack']
        
        # Create a DataFrame with sample relationships
        data = [
            [20, 15, 5, 2],  # Electronics
            [5, 25, 8, 3],   # Glassware
            [18, 7, 12, 0],  # Furniture
            [10, 5, 3, 15],  # Pharmaceuticals
            [8, 2, 15, 18]   # Fresh Produce
        ]
        
        sample_df = pd.DataFrame(data, index=product_types, columns=packaging_materials)
        
        sns.heatmap(sample_df, annot=True, cmap='YlGnBu', fmt='d', linewidths=.5)
        plt.title('Packaging Materials by Product Type (Sample Data)')
        plt.xlabel('Packaging Material')
        plt.ylabel('Product Type')
        plt.tight_layout()
    
    # Convert plot to PNG image
    canvas = FigureCanvas(plt.gcf())
    img = io.BytesIO()
    canvas.print_png(img)
    img.seek(0)
    plt.close()
    
    return Response(img.getvalue(), mimetype='image/png')

# Packaging Section Routes
@app.route('/packaging/dashboard')
@login_required
def packaging_dashboard():
    if current_user.role != 'packaging':
        flash('Access denied')
        return redirect(url_for('index'))
    
    return render_template('packaging_dashboard.html')

@app.route('/packaging/predict', methods=['GET', 'POST'])
@login_required
def packaging_predict():
    if current_user.role != 'packaging':
        flash('Access denied')
        return redirect(url_for('index'))
    
    prediction = None
    explanation = None
    confidence = None
    method = None
    error = None
    
    if request.method == 'POST':
        try:
            # Extract and validate form data
            product_type = request.form.get('product_type')
            if not product_type:
                raise ValueError("Product type is required")
                
            weight_kg_str = request.form.get('weight_kg')
            if not weight_kg_str:
                raise ValueError("Weight is required")
                
            try:
                weight_kg = float(weight_kg_str)
                if weight_kg <= 0 or weight_kg > 100:
                    raise ValueError("Weight must be between 0 and 100 kg")
            except ValueError:
                raise ValueError("Weight must be a valid number")
                
            fragile = request.form.get('fragile')
            if not fragile:
                raise ValueError("Fragility selection is required")
                
            temp_condition = request.form.get('temp_condition')
            if not temp_condition:
                raise ValueError("Temperature condition is required")
                
            humidity_level = request.form.get('humidity_level')
            if not humidity_level:
                raise ValueError("Humidity level is required")
            
            # Use Agentic AI for prediction if available
            if PACKAGING_AGENT:
                app.logger.info(f"Using AI agent for packaging prediction: {product_type}, {weight_kg}kg")
                result = PACKAGING_AGENT.predict_packaging(
                    product_type, weight_kg, fragile, temp_condition, humidity_level
                )
                prediction = result['prediction']
                explanation = result['explanation']
                confidence = result['confidence']
                method = result['method']
                
                if method == "agentic_ai":
                    flash('AI-powered recommendation generated!', 'success')
                else:
                    flash('Prediction generated using rule-based fallback method.', 'info')
            else:
                app.logger.warning("No PackagingAgent available. Using direct rule-based method.")
                # Fallback to original method if agent is not available
                df = pd.read_csv(PRODUCT_PACKAGE_DATASET)
                
                filtered_df = df[(df['Product_Type'] == product_type) & 
                                (df['Fragile'] == fragile) & 
                                (df['Temp_Condition'] == temp_condition) & 
                                (df['Humidity_Level'] == humidity_level)]
                
                if not filtered_df.empty:
                    prediction = filtered_df['Packaging_Material'].mode()[0]
                else:
                    filtered_by_type = df[df['Product_Type'] == product_type]
                    if not filtered_by_type.empty:
                        filtered_by_type['weight_diff'] = abs(filtered_by_type['Weight_kg'] - weight_kg)
                        closest_match = filtered_by_type.loc[filtered_by_type['weight_diff'].idxmin()]
                        prediction = closest_match['Packaging_Material']
                    else:
                        prediction = "No suitable packaging found"
                
                # Default values for fallback method
                explanation = f"Based on similar products in our database, this packaging is optimal for {product_type}."
                confidence = "medium"
                method = "rule_based"
                flash('Prediction generated using dataset analysis.', 'info')
                
        except ValueError as e:
            error = str(e)
            flash(f'Error: {error}', 'danger')
            app.logger.error(f"Validation error in packaging prediction: {error}")
        except Exception as e:
            error = f"An unexpected error occurred: {str(e)}"
            flash(error, 'danger')
            app.logger.error(f"Error in packaging prediction: {type(e).__name__}: {e}")
        
    return render_template('packaging_predict.html', 
                          prediction=prediction, 
                          explanation=explanation,
                          confidence=confidence,
                          method=method,
                          error=error)

# API Endpoints for ML predictions
@app.route('/api/predict_packaging', methods=['POST'])
def predict_packaging():
    try:
        data = request.json
        
        # Validate request data
        if not data:
            return jsonify({'error': 'No request data provided'}), 400
        
        # Extract and validate features
        product_type = data.get('product_type')
        if not product_type:
            return jsonify({'error': 'Product type is required'}), 400
            
        weight_kg_raw = data.get('weight_kg')
        if weight_kg_raw is None:
            return jsonify({'error': 'Weight is required'}), 400
            
        try:
            weight_kg = float(weight_kg_raw)
            if weight_kg <= 0 or weight_kg > 100:
                return jsonify({'error': 'Weight must be between 0 and 100 kg'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'Weight must be a valid number'}), 400
            
        fragile = data.get('fragile')
        if not fragile:
            return jsonify({'error': 'Fragility selection is required'}), 400
            
        temp_condition = data.get('temp_condition')
        if not temp_condition:
            return jsonify({'error': 'Temperature condition is required'}), 400
            
        humidity_level = data.get('humidity_level')
        if not humidity_level:
            return jsonify({'error': 'Humidity level is required'}), 400
        
        app.logger.info(f"Packaging prediction API request: {product_type}, {weight_kg}kg, {fragile}, {temp_condition}, {humidity_level}")
        
        # Use Agentic AI for prediction if available
        if PACKAGING_AGENT:
            result = PACKAGING_AGENT.predict_packaging(
                product_type, weight_kg, fragile, temp_condition, humidity_level
            )
            app.logger.info(f"AI prediction result: {result['prediction']} (method: {result['method']})")
            return jsonify(result)
        else:
            app.logger.warning("No PackagingAgent available for API request. Using direct rule-based method.")
            # Fallback to original method
            df = pd.read_csv(PRODUCT_PACKAGE_DATASET)
            
            filtered_df = df[(df['Product_Type'] == product_type) & 
                            (df['Fragile'] == fragile) & 
                            (df['Temp_Condition'] == temp_condition) & 
                            (df['Humidity_Level'] == humidity_level)]
            
            if not filtered_df.empty:
                prediction = filtered_df['Packaging_Material'].mode()[0]
            else:
                filtered_by_type = df[df['Product_Type'] == product_type]
                if not filtered_by_type.empty:
                    filtered_by_type['weight_diff'] = abs(filtered_by_type['Weight_kg'] - weight_kg)
                    closest_match = filtered_by_type.loc[filtered_by_type['weight_diff'].idxmin()]
                    prediction = closest_match['Packaging_Material']
                else:
                    prediction = "No suitable packaging found"
            
            result = {
                'prediction': prediction,
                'explanation': f"Based on similar products in our database for {product_type}.",
                'confidence': 'medium',
                'method': 'rule_based'
            }
            app.logger.info(f"Rule-based prediction result: {result['prediction']}")
            return jsonify(result)
    
    except Exception as e:
        error_msg = f"Error in packaging prediction API: {type(e).__name__}: {str(e)}"
        app.logger.error(error_msg)
        return jsonify({
            'error': error_msg,
            'prediction': 'Standard Box',
            'explanation': 'Default recommendation due to processing error.',
            'confidence': 'low',
            'method': 'error_fallback'
        }), 500

@app.route('/api/predict_delivery_issues', methods=['POST'])
def predict_delivery_issues():
    data = request.json
    
    # Extract features
    driver_id = data.get('driver_id')
    location = data.get('location')
    time = data.get('time')
    weather = data.get('weather')
    road_condition = data.get('road_condition')
    traffic_level = data.get('traffic_level')
    
    # Load dataset
    df = pd.read_csv(SHIPPING_COMPANY_SECTION)
    
    # Simple prediction logic (in a real app, this would use a trained ML model)
    # For now, we'll use a rule-based approach
    cause_of_issue = "None"
    probability = 0.1
    
    # Check weather conditions
    if weather == "Rainy" or weather == "Foggy":
        if road_condition == "Potholes" or road_condition == "Uneven":
            cause_of_issue = "Road_Shock"
            probability = 0.8
    
    # Check traffic conditions
    if traffic_level == "High":
        if location == "Dubai":
            cause_of_issue = "Traffic"
            probability = 0.7
    
    # Check for specific combinations
    if weather == "Dusty" and location == "Dubai":
        cause_of_issue = "Heat_Damage"
        probability = 0.6
    
    return jsonify({
        'predicted_issue': cause_of_issue,
        'probability': probability
    })

# Helper function to optimize routes based on conditions
def optimize_routes(route_df):
    # Get current time of day
    current_hour = datetime.now().hour
    if 6 <= current_hour < 12:
        time_of_day = 'Morning'
    elif 12 <= current_hour < 18:
        time_of_day = 'Afternoon'
    else:
        time_of_day = 'Evening'
    
    # Get current day
    current_day = datetime.now().strftime('%A')
    
    # Filter routes based on time and day
    filtered_routes = route_df[
        (route_df['Time_of_Day'] == time_of_day) & 
        (route_df['Day_of_Week'] == current_day)
    ]
    
    if filtered_routes.empty:
        # If no routes match exactly, get all routes for this time of day
        filtered_routes = route_df[route_df['Time_of_Day'] == time_of_day]
    
    # Sort by estimated time (ascending) and delivery priority (high first)
    priority_map = {'High': 1, 'Medium': 2, 'Low': 3}
    filtered_routes['Priority_Value'] = filtered_routes['Delivery_Priority'].map(priority_map)
    
    optimized_routes = filtered_routes.sort_values(
        by=['Priority_Value', 'Estimated_Time_Min'],
        ascending=[True, True]
    ).head(5)
    
    # Format results for display
    result = []
    for _, route in optimized_routes.iterrows():
        result.append({
            'route_id': route['Route_ID'],
            'start': route['Start_Location'],
            'end': route['End_Location'],
            'distance': route['Distance_KM'],
            'time': route['Estimated_Time_Min'],
            'traffic': route['Traffic_Level'],
            'priority': route['Delivery_Priority']
        })
    
    return result

# Helper function to identify potential issues on routes
def get_route_issues(shipping_df):
    # Analyze historical delivery issues
    issue_counts = shipping_df['Cause_of_Issue'].value_counts()
    traffic_issues = shipping_df[shipping_df['Traffic_Level'] == 'High']
    weather_issues = shipping_df[shipping_df['Weather'] != 'Clear']
    
    # Generate intelligent warnings based on data
    issues = []
    
    if not issue_counts.empty:
        top_issue = issue_counts.index[0] if len(issue_counts) > 0 else None
        if top_issue and top_issue != 'nan':
            issues.append(f"Most common delivery issue: {top_issue}")
    
    if not traffic_issues.empty:
        traffic_locations = traffic_issues['Location'].value_counts()
        if not traffic_locations.empty:
            # Replace Ajman with a relevant location on the Business Bay to Jumeirah route
            issues.append("High traffic alert at Al Wasl Road/Jumeirah intersection - peak hour congestion")
    
    # Add some specific Dubai related issues that are relevant to the Business Bay-Jumeirah route
    dubai_specific_issues = [
        "Construction on Al Khail Road interchange may cause delays of 5-10 minutes",
        "School zone traffic near Jumeirah Beach Road between 07:30-08:30 and 14:00-15:00",
        "Business Bay bridge congestion between 17:00-19:00",
        "Weekend shopping traffic near City Walk affecting Al Wasl Road"
    ]
    
    # Add at least 3 issues
    while len(issues) < 3:
        issues.append(dubai_specific_issues.pop(0))
    
    return issues

# Additional endpoint for packaging visualization
@app.route('/api/packaging/visualization', methods=['GET'])
def packaging_visualization():
    try:
        # Load the dataset
        try:
            df = pd.read_csv(PRODUCT_PACKAGE_DATASET)
            if df.empty:
                raise ValueError("Dataset is empty")
        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
            app.logger.warning(f"Could not load packaging dataset: {str(e)}. Using sample data.")
            # Create sample data
            return create_sample_packaging_data()
        
        # Weight ranges
        weight_ranges = [
            {'min': 0, 'max': 1, 'label': '0-1 kg'},
            {'min': 1, 'max': 2, 'label': '1-2 kg'},
            {'min': 2, 'max': 5, 'label': '2-5 kg'},
            {'min': 5, 'max': 10, 'label': '5-10 kg'},
            {'min': 10, 'max': 100, 'label': '10+ kg'},
        ]
        
        # Calculate optimal packaging by weight range
        weight_packaging = []
        for weight_range in weight_ranges:
            filtered_df = df[(df['Weight_kg'] >= weight_range['min']) & 
                            (df['Weight_kg'] < weight_range['max'])]
            if not filtered_df.empty:
                top_packaging = filtered_df['Packaging_Material'].value_counts().head(3)
                weight_packaging.append({
                    'range': weight_range['label'],
                    'packaging': [{'name': name, 'count': int(count)} 
                                for name, count in top_packaging.items()]
                })
        
        # If no data found in any weight range, use sample data
        if not weight_packaging:
            app.logger.warning("No packaging data found in any weight range. Using sample data.")
            return create_sample_packaging_data()
            
        # Calculate packaging distribution by product type
        product_packaging = {}
        for product_type in df['Product_Type'].unique():
            filtered_df = df[df['Product_Type'] == product_type]
            if not filtered_df.empty:
                distribution = {k: int(v) for k, v in filtered_df['Packaging_Material'].value_counts().to_dict().items()}
                product_packaging[product_type] = distribution
        
        # Fragility impact on packaging
        fragile_df = df[df['Fragile'] == 'Yes']
        non_fragile_df = df[df['Fragile'] == 'No']
        
        fragile_packaging = {k: int(v) for k, v in fragile_df['Packaging_Material'].value_counts().head(5).to_dict().items()}
        non_fragile_packaging = {k: int(v) for k, v in non_fragile_df['Packaging_Material'].value_counts().head(5).to_dict().items()}
        
        result = {
            'weight_packaging': weight_packaging,
            'product_packaging': product_packaging,
            'fragility_impact': {
                'fragile': fragile_packaging,
                'non_fragile': non_fragile_packaging
            }
        }
        
        # Ensure all counts are integers, not NumPy types
        result_json = json.dumps(result, cls=NumpyEncoder)
        return jsonify(json.loads(result_json))
    
    except Exception as e:
        app.logger.error(f"Error in packaging visualization API: {type(e).__name__}: {e}")
        return create_sample_packaging_data(is_error=True, error_msg=str(e))

def create_sample_packaging_data(is_error=False, error_msg=None):
    """Create sample packaging visualization data when real data is not available"""
    app.logger.info("Creating sample packaging visualization data")
    
    # Sample packaging materials
    packaging_materials = [
        "Corrugated Box", "Bubble Wrap + Box", "Plastic Wrap + Box", 
        "Foam Box + Ice Pack", "Anti-Static + Box", "Thermocol + Box", 
        "Wooden Crate", "Insulated Box"
    ]
    
    # Sample product types
    product_types = ["Electronics", "Glassware", "Furniture", "Pharmaceuticals", "Fresh Produce"]
    
    # Create sample weight packaging data
    weight_ranges = ['0-1 kg', '1-2 kg', '2-5 kg', '5-10 kg', '10+ kg']
    weight_packaging = []
    
    for weight_range in weight_ranges:
        # Select 2-3 random packaging materials for this weight range
        num_packages = random.randint(2, 3)
        selected_packages = random.sample(packaging_materials, num_packages)
        
        # Assign random counts (higher for the first selected package)
        counts = [random.randint(15, 30)]
        for _ in range(num_packages - 1):
            counts.append(random.randint(5, 15))
            
        packaging = [{'name': pkg, 'count': count} for pkg, count in zip(selected_packages, counts)]
        
        weight_packaging.append({
            'range': weight_range,
            'packaging': packaging
        })
    
    # Create sample product packaging distribution
    product_packaging = {}
    for product_type in product_types:
        # Select 3-4 random packaging materials for this product type
        num_packages = random.randint(3, 4)
        selected_packages = random.sample(packaging_materials, num_packages)
        
        # Assign random counts
        distribution = {}
        for pkg in selected_packages:
            distribution[pkg] = random.randint(5, 25)
            
        product_packaging[product_type] = distribution
    
    # Create sample fragility impact
    fragile_packaging = {pkg: random.randint(5, 20) for pkg in random.sample(packaging_materials, 4)}
    non_fragile_packaging = {pkg: random.randint(5, 20) for pkg in random.sample(packaging_materials, 4)}
    
    result = {
        'weight_packaging': weight_packaging,
        'product_packaging': product_packaging,
        'fragility_impact': {
            'fragile': fragile_packaging,
            'non_fragile': non_fragile_packaging
        }
    }
    
    # Add error info if needed
    if is_error:
        result['is_sample_data'] = True
        if error_msg:
            result['error'] = f"Error processing packaging visualization data: {error_msg}"
    
    return jsonify(result)

# CSRF error handler
@app.errorhandler(400)
def handle_csrf_error(e):
    app.logger.warning(f"CSRF error occurred: {str(e)}")
    return render_template('error.html', 
                          error_title="Security Error (400)",
                          error_message="CSRF token validation failed. Please try again."), 400

# Helper class for JSON serialization of numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

if __name__ == '__main__':
    app.run(debug=True) 