import os
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM as Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PackagingAgent:
    """
    Agentic AI implementation for packaging material recommendations.
    Uses RAG (Retrieval Augmented Generation) with Ollama and FAISS vector database.
    """
    
    def __init__(self, data_path, model_name="llama3.2"):
        """Initialize the PackagingAgent with dataset path and model settings"""
        self.data_path = data_path
        self.model_name = model_name
        self.embedding_model = "mxbai-embed-large"
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        
        # Initialize components
        self._init_llm()
        self._init_vectorstore()
        self._init_qa_chain()
        
        logger.info(f"PackagingAgent initialized with model: {model_name}")
    
    def _init_llm(self):
        """Initialize the Ollama LLM"""
        try:
            # Check if Ollama is available
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            
            if response.status_code == 200:
                self.llm = Ollama(model=self.model_name)
                logger.info(f"LLM initialized: {self.model_name}")
            else:
                logger.warning(f"Ollama service responded but returned error: {response.status_code}")
                self.llm = None
        except Exception as e:
            logger.warning(f"Ollama service not detected. Running in fallback mode: {e}")
            # Fallback to simple rule-based approach if LLM fails
            self.llm = None
    
    def _init_vectorstore(self):
        """Initialize the FAISS vector database from packaging dataset"""
        try:
            # Check if vectorstore already exists
            if os.path.exists("app/agentic/packaging_vectorstore"):
                self.vectorstore = FAISS.load_local(
                    "app/agentic/packaging_vectorstore",
                    OllamaEmbeddings(model=self.embedding_model)
                )
                logger.info("Loaded existing vectorstore")
            else:
                # Create documents from dataset
                logger.info("Creating new vectorstore from dataset")
                self._create_vectorstore()
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {e}")
            self.vectorstore = None
    
    def _create_vectorstore(self):
        """Create vectorstore from the packaging dataset"""
        try:
            # Load the product package dataset
            df = pd.read_csv(self.data_path)
            
            # Convert dataset to text documents for RAG
            documents = []
            for _, row in df.iterrows():
                doc_text = (
                    f"Product ID: {row['Product_ID']}\n"
                    f"Product Type: {row['Product_Type']}\n"
                    f"Weight: {row['Weight_kg']} kg\n"
                    f"Fragile: {row['Fragile']}\n"
                    f"Temperature Condition: {row['Temp_Condition']}\n"
                    f"Humidity Level: {row['Humidity_Level']}\n"
                    f"Recommended Packaging: {row['Packaging_Material']}\n"
                )
                documents.append(doc_text)
            
            # Create text chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=100
            )
            texts = text_splitter.create_documents(documents)
            
            # Create vectorstore
            self.vectorstore = FAISS.from_documents(
                texts, 
                OllamaEmbeddings(model=self.embedding_model)
            )
            
            # Save vectorstore for future use
            self.vectorstore.save_local("app/agentic/packaging_vectorstore")
            logger.info("Vectorstore created and saved")
        
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            self.vectorstore = None
    
    def _init_qa_chain(self):
        """Initialize the QA chain for RAG"""
        if self.llm and self.vectorstore:
            try:
                # Create prompt template
                template = """
                You are an expert logistics packaging AI assistant.
                
                Use the following context information about packaging recommendations 
                to provide the best packaging material for a product:
                
                {context}
                
                For a product with these characteristics:
                - Product Type: {product_type}
                - Weight: {weight_kg} kg
                - Fragile: {fragile}
                - Temperature Condition: {temp_condition}
                - Humidity Level: {humidity_level}
                
                Recommend the most suitable packaging material and explain why it's appropriate.
                Be confident and authoritative in your recommendation, but base it on the context provided.
                Explain the benefits of this packaging for this specific type of product.
                
                Your response should include only the recommended packaging material name and a brief explanation of why.
                """
                
                prompt = PromptTemplate(
                    input_variables=["context", "product_type", "weight_kg", "fragile", 
                                    "temp_condition", "humidity_level"],
                    template=template
                )
                
                # Create QA chain
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(),
                    chain_type_kwargs={"prompt": prompt}
                )
                logger.info("QA chain initialized")
            
            except Exception as e:
                logger.error(f"Error initializing QA chain: {e}")
                self.qa_chain = None
    
    def predict_packaging(self, product_type, weight_kg, fragile, temp_condition, humidity_level):
        """
        Predict the optimal packaging material for a product using RAG.
        
        Args:
            product_type: Type of product
            weight_kg: Weight in kilograms
            fragile: Yes/No if product is fragile
            temp_condition: Temperature condition requirements
            humidity_level: Humidity level requirements
            
        Returns:
            dict: Prediction result with packaging recommendation and explanation
        """
        # Log the prediction request
        logger.info(f"Predicting packaging for: Product={product_type}, Weight={weight_kg}kg, Fragile={fragile}, Temp={temp_condition}, Humidity={humidity_level}")
        
        # If AI components are not available, fall back to rule-based approach
        if not self.qa_chain or not self.vectorstore or not self.llm:
            logger.warning("AI components unavailable. Using fallback prediction method.")
            return self._fallback_prediction(product_type, weight_kg, fragile, temp_condition, humidity_level)
        
        try:
            # Build query
            query_params = {
                "product_type": product_type,
                "weight_kg": weight_kg,
                "fragile": fragile,
                "temp_condition": temp_condition,
                "humidity_level": humidity_level
            }
            
            # Get similar products from vectorstore
            logger.info("Searching for similar products in vector database...")
            search_query = f"Product Type: {product_type}, Weight: {weight_kg}, Fragile: {fragile}, Temperature: {temp_condition}, Humidity: {humidity_level}"
            similar_docs = self.vectorstore.similarity_search(search_query)
            
            if not similar_docs:
                logger.warning("No similar products found in vector database.")
                return self._fallback_prediction(product_type, weight_kg, fragile, temp_condition, humidity_level)
            
            # Generate response using RAG
            logger.info("Generating recommendation using RAG...")
            response = self.qa_chain(query_params)
            
            if not response or not response.get('result'):
                logger.warning("RAG system returned empty response.")
                return self._fallback_prediction(product_type, weight_kg, fragile, temp_condition, humidity_level)
            
            # Parse response for packaging material and explanation
            prediction_parts = response['result'].split('\n', 1)
            if len(prediction_parts) > 1:
                prediction = prediction_parts[0].strip()
                explanation = prediction_parts[1].strip()
            else:
                # If response doesn't have the expected format
                prediction_text = response['result'].strip()
                # Extract the packaging material from the response
                common_packaging = [
                    "Bubble Wrap + Box", "Foam Box + Ice Pack", "Thermocol + Box",
                    "Corrugated Box", "Insulated Box", "Plastic Wrap + Box",
                    "Bubble Wrap + Foam", "Ice Box", "Anti-Static + Box",
                    "Sealed Drum", "Wooden Crate", "Foam Insert + Box",
                    "Vacuum Sealed", "Custom Foam Case", "Sturdy Box",
                    "Molded Plastic", "Cardboard Box"
                ]
                
                for packaging in common_packaging:
                    if packaging in prediction_text:
                        prediction = packaging
                        explanation = prediction_text.replace(packaging, "").strip()
                        break
                else:
                    # If no known packaging is found in the response
                    logger.warning(f"No known packaging found in response: {prediction_text}")
                    prediction = "Custom Packaging"
                    explanation = prediction_text
            
            # Validate the prediction
            if not prediction or prediction.strip() == "":
                logger.warning("Empty prediction generated.")
                return self._fallback_prediction(product_type, weight_kg, fragile, temp_condition, humidity_level)
            
            # Return prediction
            logger.info(f"AI prediction successful: {prediction}")
            return {
                "prediction": prediction,
                "explanation": explanation,
                "confidence": "high" if len(similar_docs) > 2 else "medium",
                "method": "agentic_ai"
            }
            
        except Exception as e:
            logger.error(f"Error in predict_packaging: {type(e).__name__}: {e}")
            # Fall back to rule-based approach on error
            return self._fallback_prediction(product_type, weight_kg, fragile, temp_condition, humidity_level)
    
    def _fallback_prediction(self, product_type, weight_kg, fragile, temp_condition, humidity_level):
        """
        Fallback rule-based prediction if AI components fail.
        """
        logger.info("Using fallback rule-based prediction")
        try:
            # Load dataset for prediction
            df = pd.read_csv(self.data_path)
            
            # Simple prediction logic
            filtered_df = df[(df['Product_Type'] == product_type) & 
                            (df['Fragile'] == fragile) & 
                            (df['Temp_Condition'] == temp_condition) & 
                            (df['Humidity_Level'] == humidity_level)]
            
            if not filtered_df.empty:
                # Find the most common packaging material for the filtered rows
                prediction = filtered_df['Packaging_Material'].mode()[0]
                explanation = f"Based on similar products in our database, this packaging is optimal for {product_type} items that are {'fragile' if fragile == 'Yes' else 'not fragile'} and require {temp_condition} temperature conditions."
            else:
                # If no exact match, find the closest by product type and weight
                filtered_by_type = df[df['Product_Type'] == product_type]
                if not filtered_by_type.empty:
                    # Find closest weight
                    filtered_by_type['weight_diff'] = abs(filtered_by_type['Weight_kg'] - float(weight_kg))
                    closest_match = filtered_by_type.loc[filtered_by_type['weight_diff'].idxmin()]
                    prediction = closest_match['Packaging_Material']
                    explanation = f"Based on the closest matching {product_type} in our database with similar weight, this packaging is recommended."
                else:
                    prediction = "No suitable packaging found"
                    explanation = "We couldn't find a suitable packaging recommendation for this specific product configuration."
            
            return {
                "prediction": prediction,
                "explanation": explanation,
                "confidence": "medium",
                "method": "rule_based"
            }
            
        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
            return {
                "prediction": "Standard Box",
                "explanation": "Default recommendation due to processing error.",
                "confidence": "low",
                "method": "default"
            } 