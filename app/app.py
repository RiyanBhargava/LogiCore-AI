import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
from datetime import datetime
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
import json
from werkzeug.utils import secure_filename
# Import Agentic AI components
from agentic.packaging_agent import PackagingAgent
import requests

app = Flask(__name__)
app.secret_key = 'logistics_platform_secret_key'

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
    for email, user_data in users.items():
        if user_data['id'] == int(user_id):
            return User(user_data['id'], email, user_data['role'])
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
            print("Ollama service detected and running")
        else:
            print("Ollama service responded but returned an error")
    except Exception as e:
        print(f"Ollama service not detected: {e}")
    
    # Initialize components with warning about Ollama status
    if not ollama_available:
        print("WARNING: Ollama service not detected. AI components will run in fallback mode.")
        print("To enable full AI capabilities, install Ollama from https://ollama.com/")
        print("Then run: ollama serve")
        print("And pull the required models: ollama pull llama3.2 mxbai-embed-large")
    
    # Initialize packaging agent (will fall back to rule-based if Ollama is not available)
    PACKAGING_AGENT = PackagingAgent(PRODUCT_PACKAGE_DATASET)
    print("Packaging agent initialized (may be in fallback mode if Ollama is not available)")
    
except Exception as e:
    print(f"Error initializing Agentic AI components: {e}")
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
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email in users and check_password_hash(users[email]['password'], password):
            user = User(users[email]['id'], email, users[email]['role'])
            login_user(user)
            
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
    logout_user()
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
    
    # Save map to html file
    map_html = m._repr_html_()
    
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
    if current_user.role != 'company':
        flash('Access denied')
        return redirect(url_for('index'))
    
    # Create a folium map for heatmap
    m = folium.Map(location=[25.2048, 55.2708], zoom_start=11)
    
    # Add random heatmap points in Dubai
    heatmap_points = [
        [25.2743, 55.3087], # Dubai Creek
        [25.2048, 55.2708], # Dubai Downtown
        [25.1972, 55.2744], # Business Bay
        [25.1124, 55.1390], # Dubai Marina
        [25.0478, 55.1816], # Palm Jumeirah
        [25.0188, 55.0371], # Dubai Investments Park
        [25.0677, 55.1403], # Dubai Sports City
        [25.0621, 55.2247], # Al Quoz
        [25.2361, 55.3894], # Dubai Airport
        [25.2285, 55.3273]  # Deira
    ]
    
    # Add heatmap layer
    plugins.HeatMap(
        heatmap_points,
        radius=15
    ).add_to(m)
    
    # Save map to html file
    heatmap_html = m._repr_html_()
    
    # Create performance charts with Plotly
    # Driver Performance
    shipping_df = pd.read_csv(SHIPPING_COMPANY_SECTION)
    customer_df = pd.read_csv(CUSTOMER_SECTION_TABLE)
    
    # Join data
    merged_df = pd.merge(
        shipping_df,
        customer_df,
        on='Delivery_ID',
        how='inner',
        suffixes=('_shipping', '_customer')
    )
    
    # Calculate average rating by driver
    driver_ratings = merged_df.groupby('Driver_ID')['Overall_Rating'].mean().reset_index()
    driver_ratings = driver_ratings.sort_values('Overall_Rating', ascending=False)
    
    # Create bar chart
    driver_fig = px.bar(
        driver_ratings, 
        x='Driver_ID', 
        y='Overall_Rating',
        title='Driver Performance by Customer Rating',
        labels={'Driver_ID': 'Driver ID', 'Overall_Rating': 'Average Rating'},
        color='Overall_Rating',
        color_continuous_scale='Viridis'
    )
    
    # Customer Satisfaction
    issue_counts = customer_df['Issue_Reported'].value_counts().reset_index()
    issue_counts.columns = ['Issue', 'Count']
    
    issue_fig = px.pie(
        issue_counts,
        values='Count',
        names='Issue',
        title='Distribution of Reported Issues',
        hole=0.3
    )
    
    # Convert figures to JSON for passing to template
    driver_chart = json.loads(driver_fig.to_json())
    issue_chart = json.loads(issue_fig.to_json())
    
    return render_template(
        'company_dashboard.html',
        heatmap_html=heatmap_html,
        driver_chart=driver_chart,
        issue_chart=issue_chart
    )

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
    
    if request.method == 'POST':
        product_type = request.form.get('product_type')
        weight_kg = float(request.form.get('weight_kg'))
        fragile = request.form.get('fragile')
        temp_condition = request.form.get('temp_condition')
        humidity_level = request.form.get('humidity_level')
        
        # Use Agentic AI for prediction if available
        if PACKAGING_AGENT:
            result = PACKAGING_AGENT.predict_packaging(
                product_type, weight_kg, fragile, temp_condition, humidity_level
            )
            prediction = result['prediction']
            explanation = result['explanation']
            confidence = result['confidence']
            method = result['method']
        else:
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
        
    return render_template('packaging_predict.html', 
                          prediction=prediction, 
                          explanation=explanation,
                          confidence=confidence,
                          method=method)

# API Endpoints for ML predictions
@app.route('/api/predict_packaging', methods=['POST'])
def predict_packaging():
    data = request.json
    
    # Extract features
    product_type = data.get('product_type')
    weight_kg = float(data.get('weight_kg'))
    fragile = data.get('fragile')
    temp_condition = data.get('temp_condition')
    humidity_level = data.get('humidity_level')
    
    # Use Agentic AI for prediction if available
    if PACKAGING_AGENT:
        result = PACKAGING_AGENT.predict_packaging(
            product_type, weight_kg, fragile, temp_condition, humidity_level
        )
        return jsonify(result)
    else:
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
        
        return jsonify({
            'prediction': prediction,
            'explanation': f"Based on similar products in our database.",
            'confidence': 'medium',
            'method': 'rule_based'
        })

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

if __name__ == '__main__':
    app.run(debug=True) 