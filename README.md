# Logistics Platform with Agentic AI

A comprehensive logistics management platform that leverages Agentic AI to optimize packaging recommendations, route planning, and customer service.

![Logistics Platform](app/static/images/Agentic%20AI.webp)

## Features

- **AI-Powered Packaging Recommendations**: Get optimal packaging material suggestions based on product characteristics
- **Route Optimization**: Intelligent route planning with traffic and weather considerations
- **Delivery Issue Prediction**: Forecast potential delivery problems before they occur
- **Role-Based Access Control**: Customized interfaces for customers, drivers, packaging specialists, and company admins
- **Real-Time Analytics**: Performance dashboards with interactive visualizations

## Technical Architecture

### AI Components

- **PackagingAgent**: RAG-based AI system for packaging material recommendations

The AI component uses:
- FAISS vector database for efficient similarity search
- Ollama for LLM integration (llama3.2 by default)
- MxBai embeddings for state-of-the-art text representation
- LangChain for RAG orchestration

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd logistics-platform
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Ensure Ollama is installed and running:
   ```
   # Install Ollama from https://ollama.com/
   # Run the Ollama service
   ollama serve
   
   # Pull the required models
   ollama pull llama3.2
   ollama pull mxbai-embed-large
   ```

## Usage

1. Start the application:
   ```
   python app/app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Log in using one of the following credentials:
   - Customer: customer@example.com / customer123
   - Driver: driver@example.com / driver123
   - Packaging Specialist: packaging@example.com / packaging123
   - Company Admin: company@example.com / company123

## User Roles

### Customer
- Submit delivery feedback
- Track orders
- Chat with AI assistant

### Driver
- View optimized routes
- Report delivery issues
- Check problem areas

### Packaging Specialist
- Get AI-powered packaging recommendations
- Optimize packaging for different product types

### Company Admin
- View performance dashboards
- Monitor driver ratings
- Analyze issue reports

## Data Files

The application uses CSV files for data storage:
- `Product_Package_Dataset.csv`: Product types and recommended packaging
- `Shipping_Company_Section.csv`: Delivery data and issues
- `Customer_Section_Table.csv`: Customer feedback and ratings
- `Route_Optimization_Data.csv`: Route information with traffic conditions

## API Endpoints

- `/api/predict_packaging`: Get packaging recommendations
- `/api/predict_delivery_issues`: Predict possible delivery problems

## AI Implementation Details

### PackagingAgent

The PackagingAgent uses Retrieval Augmented Generation (RAG) to provide intelligent packaging recommendations:

1. It creates vector embeddings of product packaging data 
2. Stores these in a FAISS vector database
3. When a new product is submitted, it:
   - Finds similar products in the database
   - Uses an LLM to reason about the best packaging based on product characteristics
   - Provides a recommendation with explanation and confidence score

### Fallback Mechanism

If the AI components fail or are unavailable, the system gracefully falls back to a rule-based approach using exact or nearest-neighbor matching in the dataset.

## License

[MIT License](LICENSE)

## Acknowledgements

- Built with Flask, Pandas, LangChain, and Ollama
- Uses Folium for maps and Plotly for visualizations 
