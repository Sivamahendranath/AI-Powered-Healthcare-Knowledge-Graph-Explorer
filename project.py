import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
from dotenv import load_dotenv
import sqlite3
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
import json
import random
from datetime import date
import tempfile
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import re
import uuid
from pyvis.network import Network
import streamlit.components.v1 as components
import base64
from PIL import Image
import time
import logging
from datetime import timedelta
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

try:
    genai.configure(api_key=os.getenv("GEMINI_API"))
    client = genai.Client(api_key=os.getenv("GEMINI_API"))
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")

class GeminiRateLimiter:
    def __init__(self):
        self.requests_per_minute = 0
        self.last_reset = datetime.now()
        self.circuit_open = False
        self.failure_count = 0

    def make_request(self, model, prompt, max_retries=3):
        if self.circuit_open:
            if datetime.now() - self.last_reset > timedelta(minutes=5):
                self.circuit_open = False
                self.failure_count = 0
            else:
                raise Exception("Circuit breaker open - too many failures")
        if datetime.now() - self.last_reset > timedelta(minutes=1):
            self.requests_per_minute = 0
            self.last_reset = datetime.now()
        if self.requests_per_minute >= 8:
            wait_time = 60 - (datetime.now() - self.last_reset).seconds
            time.sleep(wait_time)
            self.requests_per_minute = 0
            self.last_reset = datetime.now()
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                self.requests_per_minute += 1
                self.failure_count = 0
                return response
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif "400" in str(e):
                    raise Exception(f"API Request Error: {str(e)}")
                else:
                    self.failure_count += 1
                    if self.failure_count >= 3:
                        self.circuit_open = True
                    if attempt == max_retries - 1:
                        raise Exception(f"API Error after {max_retries} attempts: {str(e)}")

rate_limiter = GeminiRateLimiter()

def extract_text_from_url(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(['script', 'style', 'nav', 'header', 'footer']):
            script.decompose()
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'main'])
        if main_content:
            text = ' '.join(main_content.stripped_strings)
        else:
            text = ' '.join(soup.stripped_strings)
        return text[:10000]
    except Exception as e:
        logger.error(f"Error extracting text from URL {url}: {str(e)}")
        st.error(f"Error extracting text from URL: {str(e)}")
        return ""

def search_entities(query: str, df: pd.DataFrame) -> pd.DataFrame:
    if not query or df.empty:
        return df
    query = query.lower().strip()
    query_parts = query.split()
    def search_row(row):
        try:
            if query in row['entity_name'].lower():
                return True
            if pd.notna(row['attributes']) and any(part in row['attributes'].lower() for part in query_parts):
                return True
            if pd.notna(row['relationships']) and any(part in row['relationships'].lower() for part in query_parts):
                return True
            if 'detailed_attributes' in row and pd.notna(row['detailed_attributes']):
                try:
                    attrs = json.loads(row['detailed_attributes'])
                    for value in attrs.values():
                        if isinstance(value, str) and any(part in value.lower() for part in query_parts):
                            return True
                except:
                    pass
            return False
        except Exception:
            return False
    try:
        mask = df.apply(search_row, axis=1)
        return df[mask]
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return df

def json_serial(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f'Type {type(obj)} not serializable')

def extract_relationship_type(relationship_text: str) -> str:
    if not relationship_text:
        return "related_to"
    relationship_lower = relationship_text.lower()
    
    # Enhanced hospital relationships
    healthcare_relationships = {
        'works at': 'employee',
        'employed by': 'employee',
        'affiliated with': 'affiliation',
        'practices at': 'practices_at',
        'consultant at': 'consultant',
        'head of': 'department_head',
        'director of': 'director',
        'specializes in': 'specialization',
        'located in': 'location',
        'has pharmacy': 'pharmacy_service',
        'has ambulance': 'ambulance_service',
        'available at': 'availability',
        'has address': 'address',
        'has location': 'location',
        'has service time': 'service_time',
        'has doctor': 'doctor',
        'has nurse': 'nurse',
        'has pharmacy availability': 'pharmacy_availability',
        'has ambulance service': 'ambulance_service',
        'has specialization': 'specialization',
        'has experience': 'experience',
        'has success rate': 'success_rate',
        'has availability': 'availability_time',
        'has consultation fees': 'consultation_fees',
        'has fees': 'consultation_fees',
        # Hospital-specific relationships
        'admits patients to': 'admits_to',
        'refers to': 'refers_to',
        'collaborates with': 'collaboration',
        'part of network': 'network_member',
        'has department': 'has_department',
        'has emergency services': 'has_emergency',
        'has icu': 'has_icu',
        'has operating rooms': 'has_or',
        'accredited by': 'accreditation',
        'certified by': 'certification',
        'affiliated hospital': 'hospital_affiliation',
        'teaching hospital for': 'teaching_affiliation',
        'research center': 'research_affiliation'
    }
    
    for pattern, rel_type in healthcare_relationships.items():
        if pattern in relationship_lower:
            return rel_type
    
    # Hospital-specific patterns
    hospital_patterns = [
        (r'(?:operates|runs|manages) (?:at |the )?(.+?)(?: hospital| clinic)', 'operates'),
        (r'(?:staffed by|has staff|employees include) (.+)', 'has_staff'),
        (r'(?:services include|offers|provides) (.+)', 'provides_service'),
        (r'(?:department of|division of) (.+)', 'has_department'),
        (r'(?:partner(?:ship)? with|collaborates? with) (.+)', 'partnership'),
        (r'(?:branch of|subsidiary of|part of) (.+)', 'branch_of')
    ]
    
    for pattern, rel_type in hospital_patterns:
        match = re.search(pattern, relationship_lower)
        if match:
            return rel_type
    
    if " as " in relationship_lower:
        parts = relationship_lower.split(" as ")
        if len(parts) > 1:
            return parts[-1].strip()
    
    return "related_to"

def extract_target_entity(relationship_text: str) -> str:
    if not relationship_text:
        return ""
    relationship_lower = relationship_text.lower()
    patterns = [
        (r'connected to (.*?)( as | in | via | through |$)', 1),(r'related to (.*?)( as | in | via | through |$)', 1),
        (r'works at (.*?)( as | in | via | through |$)', 1),(r'employed by (.*?)( as | in | via | through |$)', 1),
        (r'affiliated with (.*?)( as | in | via | through |$)', 1),(r'practices at (.*?)( as | in | via | through |$)', 1),
        (r'located in (.*?)( as | in | via | through |$)', 1),(r'specializes in (.*?)( as | in | via | through |$)', 1),
        (r'has address (.*?)( as | in | via | through |$)', 1),(r'has location (.*?)( as | in | via | through |$)', 1),
        (r'has service time (.*?)( as | in | via | through |$)', 1),(r'has doctor (.*?)( as | in | via | through |$)', 1),
        (r'has nurse (.*?)( as | in | via | through |$)', 1),(r'has pharmacy (.*?)( as | in | via | through |$)', 1),
        (r'has ambulance (.*?)( as | in | via | through |$)', 1),(r'has experience (.*?)( as | in | via | through |$)', 1),
        (r'has success rate (.*?)( as | in | via | through |$)', 1),(r'has consultation fees (.*?)( as | in | via | through |$)', 1),
        # Hospital-specific patterns
        (r'operates (.*?)(?: hospital| clinic)', 1),(r'staffed by (.*?)( as | in | via | through |$)', 1),
        (r'services include (.*?)( as | in | via | through |$)', 1),(r'department of (.*?)( as | in | via | through |$)', 1),
        (r'partner with (.*?)( as | in | via | through |$)', 1),(r'branch of (.*?)( as | in | via | through |$)', 1)
    ]
    for pattern, group_idx in patterns:
        match = re.search(pattern, relationship_lower)
        if match:
            target = match.group(group_idx).strip()
            target = re.sub(r'\s*(as|in|via|through)\s*$', '', target).strip()
            return target
    return ""

def normalize_entity_name(name: str) -> str:
    return re.sub(r'\s+', ' ', name.strip().lower())

def parse_ai_response(response_text: str) -> List[Dict]:
    entities_data = []
    current_entity = None
    current_data = {}
    
    # Enhanced section headers - include hospital-specific headers
    section_headers = [
        "extracted healthcare entities",
        "extracted entities", 
        "entities details and relationships",
        "entities and relationships",
        "entity details",
        "identified entities",
        "entity relationships",
        "hospital details",
        "clinic details",
        "healthcare facilities"
    ]
    
    lines = response_text.replace('\r\n', '\n').split('\n')
    entity_section_started = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this line starts a new section
        line_lower = line.lower()
        if line_lower in section_headers:
            entity_section_started = True
            continue
            
        if not entity_section_started:
            continue
            
        # Check if this is a new entity (not a bullet point or numbered list)
        if not line.startswith(('•', '-', '*', '◦', '‣')) and not re.match(r'^\d+\.', line):
            if current_data:
                entities_data.append(current_data)
            current_data = {
                'entity_name': line,
                'attributes': [],
                'relationships': [],
                'entity_type': None,
                'detailed_attributes': {}
            }
            current_entity = line
        else:
            if not current_entity:
                continue
                
            detail = re.sub(r'^[•\-\*\d\.◦‣\s]+', '', line).strip()
            
            # Enhanced relationship indicators including hospital-specific ones
            relationship_indicators = [
                'connected to', 'related to', 'linked to', 'associated with',
                'works at', 'employed by', 'affiliated with', 'practices at',
                'located in', 'specializes in', 'has pharmacy', 'has ambulance',
                'available at', 'consultant at', 'head of', 'has address',
                'has location', 'has service time', 'has doctor', 'has nurse',
                'has specialization', 'has experience', 'has success rate',
                'has consultation fees', 'has fees',
                # Hospital-specific relationships
                'admits patients to', 'refers to', 'collaborates with',
                'part of network', 'has department', 'has emergency services',
                'has icu', 'has operating rooms', 'accredited by', 'certified by',
                'affiliated hospital', 'teaching hospital for', 'research center',
                'operates', 'staffed by', 'services include', 'department of',
                'partner with', 'branch of'
            ]
            
            is_relationship = any(indicator in detail.lower() for indicator in relationship_indicators)
            
            if is_relationship:
                current_data['relationships'].append(detail)
            else:
                if ':' in detail:
                    attr_parts = detail.split(':', 1)
                    attr_name = attr_parts[0].strip().lower()
                    attr_value = attr_parts[1].strip()
                    
                    if attr_name in ['type', 'category', 'kind']:
                        current_data['entity_type'] = attr_value.lower()
                    else:
                        current_data['attributes'].append(f"{attr_name}: {attr_value}")
                        current_data['detailed_attributes'][attr_name] = attr_value
                else:
                    current_data['attributes'].append(detail)
    
    if current_data:
        entities_data.append(current_data)
    
    processed_data = []
    for idx, data in enumerate(entities_data, 1):
        if not data.get('entity_name'):
            continue
            
        # Enhanced entity type detection
        if not data.get('entity_type'):
            data['entity_type'] = detect_entity_type(data['entity_name'], data['attributes'])
        
        relationship_details = []
        associated_entities = set()
        
        for rel in data['relationships']:
            target_entity = extract_target_entity(rel)
            if target_entity:
                associated_entities.add(target_entity)
                relationship_type = extract_relationship_type(rel)
                relationship_details.append({
                    'target_entity': target_entity,
                    'relationship_type': relationship_type,
                    'full_description': rel,
                    'source_entity': data['entity_name'],
                    'source_type': data.get('entity_type', 'general')
                })
        
        clean_attributes = []
        for attr in data['attributes']:
            if attr.lower().startswith('attribute:'):
                attr = attr[len('attribute:'):].strip()
            clean_attributes.append(attr)
        
        processed_data.append({
            'entity_id': idx,
            'entity_name': data['entity_name'],
            'associated_entities': ', '.join(sorted(associated_entities)) if associated_entities else '',
            'attributes': '; '.join(clean_attributes) if clean_attributes else '',
            'relationships': '; '.join(data['relationships']) if data['relationships'] else '',
            'relationship_details': relationship_details,
            'weight': 1.0,
            'entity_type': data['entity_type'],
            'detailed_attributes': data.get('detailed_attributes', {})
        })
    
    return processed_data

def init_database():
    try:
        conn = sqlite3.connect('entities.db', timeout=30)
        cursor = conn.cursor()
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('''CREATE TABLE IF NOT EXISTS entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_name TEXT NOT NULL,
            associated_entities TEXT,
            attributes TEXT,
            relationships TEXT,
            relationship_details TEXT,
            detailed_attributes TEXT,
            weight FLOAT DEFAULT 1.0,
            entity_type TEXT DEFAULT 'general',
            frequency INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source_type TEXT,
            source_identifier TEXT,
            is_hidden BOOLEAN DEFAULT 0,
            UNIQUE(entity_name, source_type, source_identifier)
        )''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(entity_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_weight ON entities(weight)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON entities(created_at)')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS relationship_edges (
            edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER,
            target_id INTEGER,
            relationship_type TEXT,
            full_description TEXT,
            source_entity TEXT,
            source_entity_type TEXT,
            weight FLOAT DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_hidden BOOLEAN DEFAULT 0,
            UNIQUE(source_id, target_id, relationship_type),
            FOREIGN KEY(source_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
            FOREIGN KEY(target_id) REFERENCES entities(entity_id) ON DELETE CASCADE
        )''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_id ON relationship_edges(source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_target_id ON relationship_edges(target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationship_type ON relationship_edges(relationship_type)')
        
        # Create entity_attributes table for detailed attribute management
        cursor.execute('''CREATE TABLE IF NOT EXISTS entity_attributes (
            attribute_id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER,
            attribute_type TEXT,
            attribute_value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
        )''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_attributes ON entity_attributes(entity_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_attribute_type ON entity_attributes(attribute_type)')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        st.error(f"Database initialization error: {e}")

def calculate_entity_weights():
    try:
        conn = sqlite3.connect('entities.db')
        cursor = conn.cursor()
        cursor.execute('''SELECT e.entity_id,
            COUNT(DISTINCT r1.edge_id) + COUNT(DISTINCT r2.edge_id) as connection_count,
            COUNT(DISTINCT r1.source_id) as out_connections,
            COUNT(DISTINCT r2.target_id) as in_connections,
            e.frequency,
            LENGTH(e.attributes) - LENGTH(REPLACE(e.attributes, ';', '')) + 1 as attribute_count,
            e.entity_type,
            json_array_length(e.detailed_attributes) as detailed_attr_count 
            FROM entities e
            LEFT JOIN relationship_edges r1 ON e.entity_id = r1.source_id AND r1.is_hidden = 0
            LEFT JOIN relationship_edges r2 ON e.entity_id = r2.target_id AND r2.is_hidden = 0
            WHERE e.is_hidden = 0 
            GROUP BY e.entity_id''')
        
        weights_data = cursor.fetchall()
        entity_weights = {}
        
        if not weights_data:
            return
        
        max_connections = 1
        max_attributes = 1
        max_frequency = 1
        max_detailed_attrs = 1
        
        for row in weights_data:
            entity_id, connection_count, out_conn, in_conn, frequency, attribute_count, entity_type, detailed_attr_count = row
            max_connections = max(max_connections, connection_count)
            max_attributes = max(max_attributes, attribute_count if attribute_count else 0)
            max_frequency = max(max_frequency, frequency if frequency else 0)
            max_detailed_attrs = max(max_detailed_attrs, detailed_attr_count if detailed_attr_count else 0)
        
        for row in weights_data:
            entity_id, connection_count, out_conn, in_conn, frequency, attribute_count, entity_type, detailed_attr_count = row
            
            norm_connections = connection_count / max_connections if max_connections > 0 else 0
            norm_attributes = (attribute_count / max_attributes) if max_attributes > 0 and attribute_count else 0
            norm_frequency = (frequency / max_frequency) if max_frequency > 0 and frequency else 0
            norm_detailed_attrs = (detailed_attr_count / max_detailed_attrs) if max_detailed_attrs > 0 and detailed_attr_count else 0
            
            # Enhanced type multiplier for hospitals
            type_multiplier = 1.0
            if entity_type:
                entity_type_lower = entity_type.lower()
                if entity_type_lower in ['hospital', 'medical center', 'health center']:
                    type_multiplier = 2.5  # Increased for hospitals
                elif entity_type_lower in ['clinic']:
                    type_multiplier = 2.0
                elif entity_type_lower in ['doctor', 'physician', 'surgeon']:
                    type_multiplier = 1.8
                elif entity_type_lower in ['organization', 'medical_center']:
                    type_multiplier = 1.5
            
            raw_weight = ((norm_connections * 0.4) + (norm_detailed_attrs * 0.3) + 
                         (norm_attributes * 0.1) + (norm_frequency * 0.2)) * type_multiplier
            final_weight = 1 + (raw_weight * 9)
            entity_weights[entity_id] = final_weight
        
        for entity_id, weight in entity_weights.items():
            cursor.execute('UPDATE entities SET weight = ? WHERE entity_id = ?', (weight, entity_id))
        
        cursor.execute('''SELECT r.edge_id, r.source_id, r.target_id, e1.weight as source_weight, e2.weight as target_weight
            FROM relationship_edges r 
            JOIN entities e1 ON r.source_id = e1.entity_id AND e1.is_hidden = 0
            JOIN entities e2 ON r.target_id = e2.entity_id AND e2.is_hidden = 0 
            WHERE r.is_hidden = 0''')
        
        edge_data = cursor.fetchall()
        
        for row in edge_data:
            edge_id, source_id, target_id, source_weight, target_weight = row
            edge_weight = (source_weight + target_weight) / 2
            cursor.execute('UPDATE relationship_edges SET weight = ? WHERE edge_id = ?', (edge_weight, edge_id))
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error calculating entity weights: {e}")

def detect_entity_type(entity_name: str, attributes: List[str]) -> str:
    entity_name_lower = entity_name.lower()
    attributes_text = " ".join(attributes).lower()
    
    # Check attributes first
    for attr in attributes:
        if ':' in attr:
            attr_name, attr_value = attr.split(':', 1)
            attr_name = attr_name.strip().lower()
            attr_value = attr_value.strip().lower()
            if attr_name in ['type', 'category', 'kind']:
                if attr_value in ['hospital', 'clinic', 'medical center', 'health center']:
                    return attr_value
                elif attr_value in ['doctor', 'physician', 'surgeon', 'specialist']:
                    return "doctor"
    
    # Enhanced hospital detection
    hospital_indicators = [
        'hospital', 'medical center', 'health center', 'medical centre',
        'general hospital', 'memorial hospital', 'city hospital', 
        'regional hospital', 'university hospital', 'teaching hospital',
        'children\'s hospital', 'women\'s hospital', 'psychiatric hospital',
        'rehabilitation hospital', 'specialty hospital',
        'healthcare center', 'medical facility'
    ]
    
    clinic_indicators = [
        'clinic', 'medical clinic', 'health clinic', 'surgical center',
        'outpatient center', 'urgent care', 'walk-in clinic'
    ]
    
    for indicator in hospital_indicators:
        if indicator in entity_name_lower:
            return "hospital"
    
    for indicator in clinic_indicators:
        if indicator in entity_name_lower:
            return "clinic"
    
    # Check for medical facilities in attributes
    medical_facility_indicators = [
        'emergency department', 'icu', 'operating room', 'surgical',
        'patient rooms', 'wards', 'department of', 'medical staff',
        'healthcare facility', 'treatment center', 'emergency services',
        'pharmacy services', 'ambulance services', 'nursing staff'
    ]
    
    for indicator in medical_facility_indicators:
        if indicator in attributes_text:
            return "hospital"
    
    # Doctor detection
    doctor_indicators = [
        'dr.', 'doctor', 'physician', 'surgeon', 'specialist',
        'md', 'm.d.', 'mbbs', 'prof.', 'professor',
        'cardiologist', 'neurologist', 'pediatrician', 'dermatologist'
    ]
    
    for indicator in doctor_indicators:
        if indicator in entity_name_lower:
            return "doctor"
    
    # Other entity types
    if 'pharmacy' in entity_name_lower:
        return "pharmacy"
    elif 'ambulance' in entity_name_lower:
        return "ambulance_service"
    elif any(word in entity_name_lower for word in ['address', 'location', 'street', 'avenue', 'road', 'boulevard']):
        return "location"
    elif 'nurse' in entity_name_lower:
        return "nurse"
    elif any(word in entity_name_lower for word in ['service time', 'opening hours', 'closing time', 'hours of operation']):
        return "service_time"
    
    # Default based on context
    healthcare_context_words = ['medical', 'healthcare', 'patient', 'treatment', 'surgery', 
                              'emergency', 'ward', 'department', 'specialization', 'consultation',
                              'health', 'clinic', 'hospital']
    
    if any(word in attributes_text for word in healthcare_context_words):
        return "general"
    
    return "general"

def safe_json_loads(data):
    """Safely load JSON data, handling both strings and dictionaries"""
    if isinstance(data, dict):
        return data
    elif isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return {}
    else:
        return {}

def store_in_database(data: List[Dict], source_type: str, source_identifier: str) -> None:
    try:
        conn = sqlite3.connect('entities.db', timeout=30)
        cursor = conn.cursor()
        entity_name_to_id = {}
        
        for entry in data:
            entry['entity_name'] = entry['entity_name'].strip()
            if not entry['entity_name']:
                continue
                
            normalized_name = normalize_entity_name(entry['entity_name'])
            
            cursor.execute('SELECT entity_id, entity_name FROM entities WHERE LOWER(entity_name) = LOWER(?) AND source_type = ? AND source_identifier = ? LIMIT 1',
                         (entry['entity_name'], source_type, source_identifier))
            
            existing_entity = cursor.fetchone()
            
            if existing_entity:
                entity_id, existing_name = existing_entity
                cursor.execute('SELECT * FROM entities WHERE entity_id = ?', (entity_id,))
                existing_row = cursor.fetchone()
                existing_data = {
                    'entity_id': existing_row[0],
                    'entity_name': existing_row[1],
                    'associated_entities': existing_row[2] or "",
                    'attributes': existing_row[3] or "",
                    'relationships': existing_row[4] or "",
                    'relationship_details': existing_row[5] or "[]",
                    'detailed_attributes': existing_row[6] or "{}",
                    'weight': existing_row[7] or 1.0,
                    'entity_type': existing_row[8] or "general",
                    'frequency': existing_row[9] or 1
                }
                merged_data = merge_entity_details(existing_data, entry)
                
                cursor.execute('''UPDATE entities SET associated_entities = ?, attributes = ?, relationships = ?,
                    relationship_details = ?, detailed_attributes = ?, weight = ?, entity_type = ?, frequency = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE entity_id = ?''',
                    (merged_data['associated_entities'], merged_data['attributes'], merged_data['relationships'],
                     merged_data['relationship_details'], merged_data.get('detailed_attributes', '{}'), 
                     merged_data.get('weight', 1.0), merged_data['entity_type'], merged_data['frequency'], entity_id))
                
                entity_name_to_id[normalized_name] = entity_id
            else:
                entry['relationship_details'] = json.dumps(entry.get('relationship_details', []), default=json_serial)
                
                # Ensure detailed_attributes is properly formatted as JSON string
                detailed_attrs = entry.get('detailed_attributes', {})
                if isinstance(detailed_attrs, str):
                    try:
                        detailed_attrs = json.loads(detailed_attrs)
                    except (json.JSONDecodeError, TypeError):
                        detailed_attrs = {}
                entry['detailed_attributes'] = json.dumps(detailed_attrs, default=json_serial)
                
                cursor.execute('''INSERT INTO entities (entity_name, associated_entities, attributes, relationships, 
                    relationship_details, detailed_attributes, weight, entity_type, frequency, source_type, source_identifier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (entry['entity_name'], entry['associated_entities'], entry['attributes'], entry['relationships'], 
                     entry['relationship_details'], entry['detailed_attributes'], entry.get('weight', 1.0), 
                     entry.get('entity_type', 'general'), 1, source_type, source_identifier))
                
                entity_id = cursor.lastrowid
                entity_name_to_id[normalized_name] = entity_id
                
                # Store detailed attributes in entity_attributes table
                if detailed_attrs and isinstance(detailed_attrs, dict):
                    for attr_type, attr_value in detailed_attrs.items():
                        if attr_value:  # Only store non-empty values
                            cursor.execute('INSERT INTO entity_attributes (entity_id, attribute_type, attribute_value) VALUES (?, ?, ?)',
                                         (entity_id, attr_type, str(attr_value)))
        
        # Process relationships
        for entry in data:
            normalized_name = normalize_entity_name(entry['entity_name'])
            source_id = entity_name_to_id.get(normalized_name)
            if not source_id:
                continue
                
            relationship_details = []
            if 'relationship_details' in entry:
                if isinstance(entry['relationship_details'], str):
                    try:
                        relationship_details = json.loads(entry['relationship_details'])
                    except json.JSONDecodeError:
                        relationship_details = []
                else:
                    relationship_details = entry.get('relationship_details', [])
            
            for rel in relationship_details:
                if not isinstance(rel, dict):
                    continue
                    
                target_entity_name = rel.get('target_entity', '').strip()
                if not target_entity_name:
                    continue
                    
                normalized_target = normalize_entity_name(target_entity_name)
                target_id = entity_name_to_id.get(normalized_target)
                
                if not target_id:
                    cursor.execute('SELECT entity_id FROM entities WHERE LOWER(entity_name) = LOWER(?)', (target_entity_name,))
                    existing_target = cursor.fetchone()
                    if existing_target:
                        target_id = existing_target[0]
                        entity_name_to_id[normalized_target] = target_id
                    else:
                        target_type = detect_entity_type(target_entity_name, [])
                        cursor.execute('INSERT INTO entities (entity_name, entity_type, source_type, source_identifier) VALUES (?, ?, ?, ?)',
                                     (target_entity_name, target_type, source_type, source_identifier))
                        target_id = cursor.lastrowid
                        entity_name_to_id[normalized_target] = target_id
                
                relationship_type = rel.get('relationship_type', 'related to')
                full_description = rel.get('full_description', f"{entry['entity_name']} is related to {target_entity_name} as {relationship_type}")
                
                cursor.execute('SELECT edge_id FROM relationship_edges WHERE source_id = ? AND target_id = ? AND relationship_type = ?',
                             (source_id, target_id, relationship_type))
                existing_edge = cursor.fetchone()
                
                if not existing_edge:
                    cursor.execute('''INSERT INTO relationship_edges (source_id, target_id, relationship_type, full_description, 
                        source_entity, source_entity_type, weight) VALUES (?, ?, ?, ?, ?, ?, ?)''',
                        (source_id, target_id, relationship_type, full_description, entry['entity_name'], entry.get('entity_type', 'general'), 1.0))
        
        conn.commit()
        conn.close()
        calculate_entity_weights()
        
    except Exception as e:
        logger.error(f"Error storing in database: {e}")
        st.error(f"Database storage error: {e}")

def merge_entity_details(existing: Dict, new: Dict) -> Dict:
    merged = existing.copy()
    
    existing_associates = set(existing['associated_entities'].split(', ')) if existing['associated_entities'] else set()
    new_associates = set(new['associated_entities'].split(', ')) if new['associated_entities'] else set()
    merged['associated_entities'] = ', '.join(sorted(existing_associates | new_associates - {''}))
    
    existing_attrs = set(existing['attributes'].split('; ')) if existing['attributes'] else set()
    new_attrs = set(new['attributes'].split('; ')) if new['attributes'] else set()
    merged['attributes'] = '; '.join(sorted(existing_attrs | new_attrs - {''}))
    
    existing_rels = set(existing['relationships'].split('; ')) if existing['relationships'] else set()
    new_rels = set(new['relationships'].split('; ')) if new['relationships'] else set()
    merged['relationships'] = '; '.join(sorted(existing_rels | new_rels - {''}))
    
    existing_rel_details = []
    if existing['relationship_details']:
        try:
            existing_rel_details = json.loads(existing['relationship_details'])
        except json.JSONDecodeError:
            pass
    
    new_rel_details = new.get('relationship_details', [])
    existing_rel_lookup = {}
    
    for detail in existing_rel_details:
        if isinstance(detail, dict) and 'target_entity' in detail and 'relationship_type' in detail:
            key = (detail['target_entity'].lower(), detail['relationship_type'].lower())
            existing_rel_lookup[key] = detail
    
    for detail in new_rel_details:
        if isinstance(detail, dict) and 'target_entity' in detail and 'relationship_type' in detail:
            key = (detail['target_entity'].lower(), detail['relationship_type'].lower())
            if key not in existing_rel_lookup:
                existing_rel_details.append(detail)
    
    merged['relationship_details'] = json.dumps(existing_rel_details, default=json_serial)
    
    # Handle detailed_attributes merging safely
    existing_detailed_attrs = safe_json_loads(existing.get('detailed_attributes', '{}'))
    new_detailed_attrs = safe_json_loads(new.get('detailed_attributes', {}))
    merged_detailed_attrs = {**existing_detailed_attrs, **new_detailed_attrs}
    merged['detailed_attributes'] = json.dumps(merged_detailed_attrs, default=json_serial)
    
    existing_freq = int(existing.get('frequency', 1)) if str(existing.get('frequency', 1)).isdigit() else 1
    merged['frequency'] = existing_freq + 1
    
    if new.get('entity_type') and new['entity_type'].lower() != 'general':
        merged['entity_type'] = new['entity_type']
    
    return merged

def process_input_with_model(text: str, source_type: str, source_identifier: str, model_name: str = "gemini-2.5-flash"):
    prompt = """Extract healthcare entities from the following text, with particular focus on HOSPITALS, clinics, doctors, and related healthcare services. Extract the following information:

CRITICAL INSTRUCTIONS - HOSPITALS AND CLINICS ARE HIGH PRIORITY:
- ALWAYS extract hospitals and clinics as primary entities
- NEVER filter out hospital names even if they seem generic
- Extract ALL healthcare facilities mentioned in the text

FOR HOSPITALS AND CLINICS - EXTRACT THESE DETAILS:
• Name of hospital/clinic (EXTRACT ALL NAMES MENTIONED)
• Type: Hospital/Clinic/Medical Center
• Location/Address: [Full address, city, area, street]
• Service Hours: [EXACT Opening hours - Closing hours, e.g., "9:00 AM - 6:00 PM", "24/7"]
• Pharmacy Services: [Yes/No/Details, e.g., "24-hour pharmacy", "on-site pharmacy"]
• Ambulance Services: [Yes/No/Details, e.g., "emergency ambulance", "air ambulance"]
• Nurses Availability: [Details, e.g., "registered nurses", "nursing staff 24/7"]
• Associated Doctors: [List of doctors affiliated with this hospital]
• Specializations/Services: [e.g., "cardiology department", "emergency care", "surgical services"]
• Contact Information: [Phone numbers, email, website if available]
• Accreditation/Certifications: [e.g., "JCI accredited", "ISO certified"]

FOR DOCTORS:
• Name of doctor
• Specialization/Medical field
• Years of experience (extract exact number if mentioned)
• Success rates (extract exact percentage if mentioned)
• Consultation fees (extract exact amount if mentioned)
• Availability (specific schedule, days, hours)
• Associated hospital/clinic

FOR ADDRESS/LOCATION:
• Full address details
• City, state, zip code
• Geographical coordinates if available

FOR SERVICE TIME:
• Opening hours (specific times)
• Closing hours (specific times)
• Days of operation
• Emergency services availability

Format your response exactly as follows:

Extracted Healthcare Entities
• [Hospital/Clinic Name 1]
• [Hospital/Clinic Name 2]
• [Doctor Name]
• [Address/Location]
• [Service Time]

Entities Details and Relationships
[Hospital/Clinic Name 1]
• Type: Hospital
• Location: [Address, city, area]
• Service Hours: [EXACT Opening hours - Closing hours]
• Pharmacy: [Yes/No/Details]
• Ambulance: [Yes/No/Details]
• Nurses: [Availability details]
• Specializations: [List of medical specialties available]
• Contact: [Phone, email, website]
• Connected to [Doctor Name] as [employed doctor/affiliated doctor/surgeon]
• Connected to [Address] as [has address/located at]
• Connected to [Service Time] as [has service time/operating hours]

[Hospital/Clinic Name 2]
• Type: Clinic
• Location: [Address, city, area]
• Service Hours: [EXACT Opening hours - Closing hours]
• Pharmacy: [Yes/No/Details]
• Ambulance: [Yes/No/Details]
• Nurses: [Availability details]
• Connected to [Doctor Name] as [consultant at/visiting doctor]

[Doctor Name]
• Type: Doctor
• Specialization: [Medical specialization]
• Experience: [Years of experience - be specific]
• Success Rate: [Exact percentage if available]
• Consultation Fees: [Exact fee information]
• Availability: [EXACT Schedule information]
• Connected to [Hospital/Clinic Name] as [works at/affiliated with/consultant at]

[Address/Location]
• Type: Location
• Full Address: [Complete address details]
• City: [City name]
• State: [State name]
• Connected to [Hospital/Clinic Name] as [location of/address for]

[Service Time]
• Type: Service_Time
• Opening Hours: [SPECIFIC opening time]
• Closing Hours: [SPECIFIC closing time]
• Days: [Days of operation]
• Emergency Services: [Yes/No]

IMPORTANT GUIDELINES:
1. EXTRACT ALL HOSPITALS AND CLINICS mentioned in the text
2. If multiple hospitals/clinics are mentioned, extract each one separately
3. For hospitals, extract ALL available details: location, services, hours, contact info
4. Look for hospital indicators: "General Hospital", "Medical Center", "Health Center", "Memorial Hospital", "City Hospital", "Regional Hospital"
5. Extract hospital networks: e.g., "Mayo Clinic", "Cleveland Clinic", "Johns Hopkins Hospital"
6. Include teaching hospitals, research hospitals, specialty hospitals
7. Extract hospital departments: "emergency department", "ICU", "surgical ward", "cardiology unit"
8. Always create relationships between hospitals and doctors
9. Extract phone numbers, websites, and contact information for hospitals
10. If a hospital has multiple locations, extract each location separately

Text: """ + text[:8000]
    
    try:
        model = genai.GenerativeModel(model_name)
        response = rate_limiter.make_request(model, prompt)
        
        if response and response.text:
            entities_data = parse_ai_response(response.text)
            
            if entities_data:
                store_in_database(entities_data, source_type, source_identifier)
                display_data = []
                
                for item in entities_data:
                    display_item = {
                        'entity_name': item['entity_name'],
                        'entity_type': item['entity_type'],
                        'attributes': item['attributes'],
                        'relationships': item['relationships'],
                        'associated_entities': item['associated_entities'],
                        'weight': item['weight']
                    }
                    display_data.append(display_item)
                
                display_df = pd.DataFrame(display_data)
                
                # Count hospitals and clinics
                hospital_count = len([item for item in entities_data if item['entity_type'] in ['hospital', 'clinic']])
                
                st.success(f"✅ Successfully extracted {len(entities_data)} healthcare entities! ({hospital_count} hospitals/clinics)")
                
                st.subheader("📋 Extracted Healthcare Information")
                
                def color_entity_type(val):
                    color_map = {
                        'hospital': '#1f77b4',
                        'clinic': '#1f77b4',
                        'doctor': '#ff7f0e',
                        'pharmacy': '#2ca02c',
                        'ambulance_service': '#d62728',
                        'person': '#9467bd',
                        'organization': '#8c564b',
                        'location': '#e377c2',
                        'general': '#7f7f7f',
                        'nurse': '#17becf',
                        'service_time': '#bcbd22',
                        'address': '#7f7f7f'
                    }
                    color = color_map.get(val.lower(), '#7f7f7f')
                    return f'background-color: {color}; color: white;'
                
                try:
                    styled_df = display_df.style.map(color_entity_type, subset=['entity_type'])
                    st.dataframe(styled_df, hide_index=True, use_container_width=True)
                except Exception as e:
                    st.dataframe(display_df, hide_index=True, use_container_width=True)
                
                # Show hospital-specific summary
                if hospital_count > 0:
                    hospitals = [item for item in entities_data if item['entity_type'] in ['hospital', 'clinic']]
                    st.subheader("🏥 Extracted Hospitals/Clinics")
                    for hospital in hospitals[:5]:  # Show first 5
                        st.markdown(f"""
                        <div style="background-color: #e8f4fc; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #1f77b4;">
                            <strong>{hospital['entity_name']}</strong> ({hospital['entity_type'].title()})<br>
                            <small>Attributes: {len(hospital['attributes'].split(';')) if hospital['attributes'] else 0} | 
                            Relationships: {len(hospital['relationships'].split(';')) if hospital['relationships'] else 0}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"entities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"download_csv_{uuid.uuid4()}"
                )
                return True
    
    except Exception as e:
        logger.error(f"Error in process_input_with_model: {e}")
        st.error(f"❌ Error processing input: {str(e)}")
        
        if "400" in str(e):
            st.info("💡 API model error. Trying fallback model...")
            try:
                return process_input_with_model(text, source_type, source_identifier, "gemini-2.5-flash")
            except:
                st.error("Please check your API key and model availability")
        elif "429" in str(e):
            st.info("💡 Rate limit exceeded. Please wait a minute and try again.")
        elif "401" in str(e):
            st.error("🔑 Invalid API key. Please check your Gemini API key configuration.")
    
    return False

def process_input(text: str, source_type: str, source_identifier: str):
    return process_input_with_model(text, source_type, source_identifier, "gemini-2.5-flash")

def get_stored_entities() -> pd.DataFrame:
    try:
        conn = sqlite3.connect('entities.db', timeout=30)
        df = pd.read_sql_query('''SELECT entity_id, entity_name, associated_entities, attributes, 
            relationships, relationship_details, detailed_attributes, weight, entity_type, frequency, created_at, 
            updated_at, source_type, source_identifier FROM entities WHERE is_hidden = 0
            ORDER BY weight DESC, updated_at DESC''', conn)
        
        if not df.empty and 'entity_id' in df.columns:
            df['entity_id'] = df['entity_id'].astype(int)
        
        conn.close()
        return df
    
    except Exception as e:
        logger.error(f"Error retrieving stored entities: {e}")
        st.error(f"Error retrieving data: {e}")
        return pd.DataFrame()

def get_relationship_edges() -> pd.DataFrame:
    try:
        conn = sqlite3.connect('entities.db', timeout=30)
        df = pd.read_sql_query('''SELECT e.edge_id, e.source_id, s.entity_name as source_name, 
            e.target_id, t.entity_name as target_name, e.relationship_type, e.full_description, e.weight,
            e.source_entity, e.source_entity_type FROM relationship_edges e 
            JOIN entities s ON e.source_id = s.entity_id JOIN entities t ON e.target_id = t.entity_id
            WHERE e.is_hidden = 0 AND s.is_hidden = 0 AND t.is_hidden = 0 ORDER BY e.weight DESC''', conn)
        conn.close()
        return df
    
    except Exception as e:
        logger.error(f"Error retrieving relationship edges: {e}")
        return pd.DataFrame()

def get_entity_attributes(entity_id: int) -> pd.DataFrame:
    try:
        conn = sqlite3.connect('entities.db', timeout=30)
        df = pd.read_sql_query('''SELECT attribute_id, attribute_type, attribute_value, created_at 
            FROM entity_attributes WHERE entity_id = ? ORDER BY attribute_type''', conn, params=(entity_id,))
        conn.close()
        return df
    
    except Exception as e:
        logger.error(f"Error retrieving entity attributes: {e}")
        return pd.DataFrame()

def find_all_paths(edges_df: pd.DataFrame, start_id: int, end_id: int, max_depth: int = 4) -> List[List[int]]:
    graph = {}
    for _, edge in edges_df.iterrows():
        source_id = edge['source_id']
        target_id = edge['target_id']
        if source_id not in graph:
            graph[source_id] = []
        graph[source_id].append(target_id)
    
    paths = []
    
    def dfs(current_id, current_path, visited):
        if len(current_path) > max_depth:
            return
        if current_id == end_id:
            paths.append(current_path.copy())
            return
        if current_id not in graph:
            return
        
        for next_id in graph[current_id]:
            if next_id not in visited:
                visited.add(next_id)
                current_path.append(next_id)
                dfs(next_id, current_path, visited)
                current_path.pop()
                visited.remove(next_id)
    
    visited = {start_id}
    dfs(start_id, [start_id], visited)
    return paths

def get_entity_connections(entity_id: int, edges_df: pd.DataFrame) -> Dict:
    connections = {'outgoing': [], 'incoming': []}
    
    for _, edge in edges_df.iterrows():
        if edge['source_id'] == entity_id:
            connections['outgoing'].append({
                'target_id': edge['target_id'],
                'target_name': edge['target_name'],
                'relationship_type': edge['relationship_type'],
                'full_description': edge['full_description'],
                'weight': edge['weight'],
                'source_entity_type': edge.get('source_entity_type', 'general')
            })
        elif edge['target_id'] == entity_id:
            connections['incoming'].append({
                'source_id': edge['source_id'],
                'source_name': edge['source_name'],
                'relationship_type': edge['relationship_type'],
                'full_description': edge['full_description'],
                'weight': edge['weight'],
                'source_entity_type': edge.get('source_entity_type', 'general')
            })
    
    connections['outgoing'] = sorted(connections['outgoing'], key=lambda x: x['weight'], reverse=True)
    connections['incoming'] = sorted(connections['incoming'], key=lambda x: x['weight'], reverse=True)
    
    return connections

def display_entity_details(entity_id: int, stored_df: pd.DataFrame):
    """Display detailed information about an entity in a structured format"""
    entity_data = stored_df[stored_df['entity_id'] == entity_id]
    if entity_data.empty:
        st.warning("Entity not found")
        return
    
    entity_row = entity_data.iloc[0]
    entity_name = entity_row['entity_name']
    entity_type = entity_row['entity_type']
    
    # Parse detailed attributes
    detailed_attrs = safe_json_loads(entity_row.get('detailed_attributes', '{}'))
    
    # Create a beautiful card-like display
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {'#1f77b4' if entity_type == 'hospital' else '#ff7f0e' if entity_type == 'doctor' else '#2ca02c'}, #6e8efb); 
        padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h3 style="margin: 0; color: white;">{entity_name}</h3>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">
            {entity_type.capitalize()} • Weight: {entity_row['weight']:.2f}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display based on entity type
    if entity_type == 'doctor':
        display_doctor_details(entity_row, detailed_attrs)
    elif entity_type in ['hospital', 'clinic']:
        display_hospital_details(entity_row, detailed_attrs)
    elif entity_type == 'location':
        display_location_details(entity_row, detailed_attrs)
    elif entity_type == 'service_time':
        display_service_time_details(entity_row, detailed_attrs)
    else:
        display_general_details(entity_row, detailed_attrs)
    
    # Display relationships
    edges_df = get_relationship_edges()
    connections = get_entity_connections(entity_id, edges_df)
    
    if connections['outgoing'] or connections['incoming']:
        st.subheader("🔗 Connections")
        
        if connections['outgoing']:
            st.markdown("**Outgoing Connections:**")
            for conn in connections['outgoing'][:5]:  # Show top 5
                st.markdown(f"""
                <div style="background-color: #e8f4fc; padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem; border-left: 3px solid #1f77b4;">
                    <b>→ {conn['target_name']}</b><br>
                    <small>Relationship: {conn['relationship_type'].replace('_', ' ').title()}</small>
                </div>
                """, unsafe_allow_html=True)
        
        if connections['incoming']:
            st.markdown("**Incoming Connections:**")
            for conn in connections['incoming'][:5]:  # Show top 5
                st.markdown(f"""
                <div style="background-color: #fff3cd; padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem; border-left: 3px solid #ff7f0e;">
                    <b>← {conn['source_name']}</b><br>
                    <small>Relationship: {conn['relationship_type'].replace('_', ' ').title()}</small>
                </div>
                """, unsafe_allow_html=True)

def display_doctor_details(entity_row, detailed_attrs):
    """Display doctor-specific details"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Specialization
        specialization = detailed_attrs.get('specialization', 'Not specified')
        st.metric("Specialization", specialization)
        
        # Experience
        experience = detailed_attrs.get('experience', 'Not specified')
        st.metric("Experience", experience)
        
        # Success Rate
        success_rate = detailed_attrs.get('success rate', detailed_attrs.get('success_rate', 'Not specified'))
        if success_rate != 'Not specified':
            st.metric("Success Rate", success_rate)
    
    with col2:
        # Consultation Fees
        fees = detailed_attrs.get('consultation fees', detailed_attrs.get('fees', 'Not specified'))
        st.metric("Consultation Fees", fees)
        
        # Availability
        availability = detailed_attrs.get('availability', 'Not specified')
        if availability != 'Not specified':
            st.markdown(f"""
            <div style="background-color: #d4edda; padding: 0.75rem; border-radius: 5px; border: 1px solid #c3e6cb; margin-bottom: 1rem;">
                <strong>Availability:</strong><br>
                {availability}
            </div>
            """, unsafe_allow_html=True)
        
        # Extract from attributes if not in detailed_attrs
        attributes = entity_row['attributes'].split('; ') if entity_row['attributes'] else []
        for attr in attributes:
            if ':' in attr:
                key, value = attr.split(':', 1)
                key_lower = key.strip().lower()
                if key_lower not in detailed_attrs and key_lower in ['experience', 'specialization', 'fees', 'availability']:
                    st.metric(key.strip().title(), value.strip())

def display_hospital_details(entity_row, detailed_attrs):
    """Display hospital/clinic-specific details"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏥 Hospital Information**")
        
        # Location
        location = detailed_attrs.get('location', 'Not specified')
        st.metric("📍 Location", location)
        
        # Service Hours
        service_hours = detailed_attrs.get('service hours', detailed_attrs.get('service_hours', 'Not specified'))
        if service_hours != 'Not specified':
            st.markdown(f"""
            <div style="background-color: #d1ecf1; padding: 0.75rem; border-radius: 5px; border: 1px solid #bee5eb; margin-bottom: 1rem;">
                <strong>⏰ Service Hours:</strong><br>{service_hours}
            </div>
            """, unsafe_allow_html=True)
        
        # Contact Information
        contact = detailed_attrs.get('contact', detailed_attrs.get('phone', 'Not specified'))
        if contact != 'Not specified':
            st.metric("📞 Contact", contact)
    
    with col2:
        st.markdown("**🩺 Services & Facilities**")
        
        # Pharmacy
        pharmacy = detailed_attrs.get('pharmacy', 'Not specified')
        if pharmacy.lower() != 'not specified':
            color = "green" if pharmacy.lower() == 'yes' else "red" if pharmacy.lower() == 'no' else "orange"
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 0.5rem; border-radius: 5px; border: 1px solid {color}; margin-bottom: 0.5rem;">
                <strong>💊 Pharmacy:</strong> {pharmacy}
            </div>
            """, unsafe_allow_html=True)
        
        # Ambulance
        ambulance = detailed_attrs.get('ambulance', 'Not specified')
        if ambulance.lower() != 'not specified':
            color = "green" if ambulance.lower() == 'yes' else "red" if ambulance.lower() == 'no' else "orange"
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 0.5rem; border-radius: 5px; border: 1px solid {color}; margin-bottom: 0.5rem;">
                <strong>🚑 Ambulance:</strong> {ambulance}
            </div>
            """, unsafe_allow_html=True)
        
        # Nurses
        nurses = detailed_attrs.get('nurses', 'Not specified')
        if nurses != 'Not specified':
            st.metric("👩‍⚕️ Nurses", nurses)
        
        # Specializations
        specializations = detailed_attrs.get('specializations', detailed_attrs.get('services', 'Not specified'))
        if specializations != 'Not specified':
            st.markdown(f"""
            <div style="background-color: #e8f4fc; padding: 0.5rem; border-radius: 5px; border: 1px solid #1f77b4; margin-top: 0.5rem;">
                <strong>🎯 Specializations:</strong><br>{specializations}
            </div>
            """, unsafe_allow_html=True)

def display_location_details(entity_row, detailed_attrs):
    """Display location-specific details"""
    # Full Address
    full_address = detailed_attrs.get('full address', detailed_attrs.get('full_address', 'Not specified'))
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; border: 1px solid #e9ecef; margin-bottom: 1rem;">
        <strong>Full Address:</strong><br>
        {full_address}
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        city = detailed_attrs.get('city', 'Not specified')
        st.metric("City", city)
    
    with col2:
        state = detailed_attrs.get('state', 'Not specified')
        st.metric("State", state)
    
    with col3:
        # Try to extract from attributes
        attributes = entity_row['attributes'].split('; ') if entity_row['attributes'] else []
        for attr in attributes:
            if 'zip' in attr.lower() or 'postal' in attr.lower():
                if ':' in attr:
                    _, value = attr.split(':', 1)
                    st.metric("Zip Code", value.strip())
                    break

def display_service_time_details(entity_row, detailed_attrs):
    """Display service time-specific details"""
    col1, col2 = st.columns(2)
    
    with col1:
        opening_hours = detailed_attrs.get('opening hours', detailed_attrs.get('opening_hours', 'Not specified'))
        st.markdown(f"""
        <div style="background-color: #d4edda; padding: 0.75rem; border-radius: 5px; border: 1px solid #c3e6cb; margin-bottom: 1rem;">
            <strong>Opening Hours:</strong><br>
            {opening_hours}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        closing_hours = detailed_attrs.get('closing hours', detailed_attrs.get('closing_hours', 'Not specified'))
        st.markdown(f"""
        <div style="background-color: #f8d7da; padding: 0.75rem; border-radius: 5px; border: 1px solid #f5c6cb; margin-bottom: 1rem;">
            <strong>Closing Hours:</strong><br>
            {closing_hours}
        </div>
        """, unsafe_allow_html=True)
    
    # Days of operation
    days = detailed_attrs.get('days', 'Not specified')
    if days != 'Not specified':
        st.markdown(f"""
    <div style="background-color: #fff3cd; padding: 0.75rem; border-radius: 5px; border: 1px solid #ffeaa7; margin-bottom: 1rem;">
        <strong>Days of Operation:</strong><br>
        {days}
    </div>
    """, unsafe_allow_html=True)
    
    # Emergency services
    emergency = detailed_attrs.get('emergency services', detailed_attrs.get('emergency_services', 'Not specified'))
    if emergency.lower() != 'not specified':
        color = "green" if emergency.lower() == 'yes' else "red" if emergency.lower() == 'no' else "orange"
        st.markdown(f"""
    <div style="background-color: {color}20; padding: 0.75rem; border-radius: 5px; border: 1px solid {color}; margin-bottom: 1rem;">
        <strong>Emergency Services:</strong> {emergency}
    </div>
    """, unsafe_allow_html=True)

def display_general_details(entity_row, detailed_attrs):
    """Display general entity details"""
    st.markdown("**Attributes:**")
    if detailed_attrs:
        for key, value in detailed_attrs.items():
            if value and value != 'Not specified':
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem;">
                    <strong>{key.replace('_', ' ').title()}:</strong> {value}
                </div>
                """, unsafe_allow_html=True)
    
    # Also show raw attributes
    attributes = entity_row['attributes'].split('; ') if entity_row['attributes'] else []
    if attributes:
        st.markdown("**Additional Information:**")
        for attr in attributes:
            if ':' in attr:
                key, value = attr.split(':', 1)
                key_lower = key.strip().lower()
                if key_lower not in detailed_attrs and value.strip():
                    st.markdown(f"""
                    <div style="background-color: #e9ecef; padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem;">
                        <strong>{key.strip().title()}:</strong> {value.strip()}
                    </div>
                    """, unsafe_allow_html=True)

def initialize_graph_state():
    if 'graph_nodes' not in st.session_state:
        st.session_state.graph_nodes = set()
    if 'graph_edges' not in st.session_state:
        st.session_state.graph_edges = set()
    if 'expanded_nodes' not in st.session_state:
        st.session_state.expanded_nodes = set()
    if 'node_colors' not in st.session_state:
        st.session_state.node_colors = {}
    if 'entity_types' not in st.session_state:
        st.session_state.entity_types = {}
    if 'hierarchy_level' not in st.session_state:
        st.session_state.hierarchy_level = {}
    if 'node_parents' not in st.session_state:
        st.session_state.node_parents = {}
    if 'expanded_by_type' not in st.session_state:
        st.session_state.expanded_by_type = {}
    if 'selected_node' not in st.session_state:
        st.session_state.selected_node = None
    if 'relationship_filter' not in st.session_state:
        st.session_state.relationship_filter = None
    if 'graph_options' not in st.session_state:
        st.session_state.graph_options = {
            'node_size_multiplier': 5,
            'edge_width_multiplier': 0.5,
            'color_scheme': 'default',
            'physics_enabled': True,
            'hierarchical_layout': False,
            'cluster_nodes': True,
            'show_edge_labels': True,
            'show_node_labels': True,
            'show_attribute_details': True,
            'expansion_limit': 5,
            'dark_mode': False,
            'node_shape': 'dot',
            'edge_smoothness': 'continuous',
            'font_size': 12,
            'show_attribute_values': True
        }
    if 'node_visibility' not in st.session_state:
        st.session_state.node_visibility = {}
    if 'edge_visibility' not in st.session_state:
        st.session_state.edge_visibility = {}
    if 'targeted_expansion' not in st.session_state:
        st.session_state.targeted_expansion = {}
    if 'continuous_expansion' not in st.session_state:
        st.session_state.continuous_expansion = {}
    if 'hierarchical_expansion' not in st.session_state:
        st.session_state.hierarchical_expansion = {}

def get_entity_type_color(entity_type):
    color_map = {
        'hospital': '#1f77b4',
        'clinic': '#1f77b4',
        'doctor': '#ff7f0e',
        'pharmacy': '#2ca02c',
        'ambulance_service': '#d62728',
        'person': '#9467bd',
        'organization': '#8c564b',
        'location': '#e377c2',
        'general': '#7f7f7f',
        'nurse': '#17becf',
        'service_time': '#bcbd22',
        'address': '#7f7f7f',
        'specialization': '#ff7f0e',
        'experience': '#ff7f0e',
        'success_rate': '#ff7f0e',
        'consultation_fees': '#ff7f0e',
        'availability': '#ff7f0e',
        'location_attr': '#e377c2',
        'service_hours': '#bcbd22'
    }
    return color_map.get(entity_type.lower(), '#7f7f7f')

def get_related_entities_by_type(entity_id, entity_type, connection_direction="both"):
    edges_df = get_relationship_edges()
    stored_df = get_stored_entities()
    related_entities = []
    
    if connection_direction in ["outgoing", "both"]:
        outgoing = edges_df[edges_df['source_id'] == entity_id]
        for _, edge in outgoing.iterrows():
            target_id = edge['target_id']
            target_data = stored_df[stored_df['entity_id'] == target_id]
            if not target_data.empty and target_data.iloc[0]['entity_type'].lower() == entity_type.lower():
                related_entities.append({
                    'entity_id': target_id,
                    'entity_name': edge['target_name'],
                    'entity_type': target_data.iloc[0]['entity_type'],
                    'weight': target_data.iloc[0]['weight'],
                    'relationship': edge['relationship_type'],
                    'direction': 'outgoing'
                })
    
    if connection_direction in ["incoming", "both"]:
        incoming = edges_df[edges_df['target_id'] == entity_id]
        for _, edge in incoming.iterrows():
            source_id = edge['source_id']
            source_data = stored_df[stored_df['entity_id'] == source_id]
            if not source_data.empty and source_data.iloc[0]['entity_type'].lower() == entity_type.lower():
                related_entities.append({
                    'entity_id': source_id,
                    'entity_name': edge['source_name'],
                    'entity_type': source_data.iloc[0]['entity_type'],
                    'weight': source_data.iloc[0]['weight'],
                    'relationship': edge['relationship_type'],
                    'direction': 'incoming'
                })
    
    return related_entities

def get_entity_types_from_connections(entity_id):
    edges_df = get_relationship_edges()
    stored_df = get_stored_entities()
    entity_types = set()
    
    outgoing = edges_df[edges_df['source_id'] == entity_id]
    for _, edge in outgoing.iterrows():
        target_id = edge['target_id']
        target_data = stored_df[stored_df['entity_id'] == target_id]
        if not target_data.empty:
            entity_types.add(target_data.iloc[0]['entity_type'])
    
    incoming = edges_df[edges_df['target_id'] == entity_id]
    for _, edge in incoming.iterrows():
        source_id = edge['source_id']
        source_data = stored_df[stored_df['entity_id'] == source_id]
        if not source_data.empty:
            entity_types.add(source_data.iloc[0]['entity_type'])
    
    return list(entity_types)

def expand_node_by_type(entity_id, entity_type):
    if entity_id not in st.session_state.expanded_by_type:
        st.session_state.expanded_by_type[entity_id] = set()
    
    if entity_type in st.session_state.expanded_by_type[entity_id]:
        st.session_state.expanded_by_type[entity_id].remove(entity_type)
        return
    
    st.session_state.expanded_by_type[entity_id].add(entity_type)
    related_entities = get_related_entities_by_type(entity_id, entity_type)
    stored_df = get_stored_entities()
    expansion_limit = st.session_state.graph_options.get('expansion_limit', 5)
    added_count = 0
    
    for entity in related_entities:
        if added_count >= expansion_limit:
            break
        
        related_id = entity['entity_id']
        if related_id not in st.session_state.graph_nodes:
            st.session_state.graph_nodes.add(related_id)
            related_data = stored_df[stored_df['entity_id'] == related_id]
            if not related_data.empty:
                st.session_state.entity_types[related_id] = related_data.iloc[0]['entity_type']
                st.session_state.node_colors[related_id] = get_entity_type_color(related_data.iloc[0]['entity_type'])
                st.session_state.node_visibility[related_id] = True
        
        if entity['direction'] == 'outgoing':
            edge = (entity_id, related_id, entity['relationship'])
        else:
            edge = (related_id, entity_id, entity['relationship'])
        
        st.session_state.graph_edges.add(edge)
        st.session_state.edge_visibility[edge] = True
        
        if related_id not in st.session_state.node_parents:
            st.session_state.node_parents[related_id] = set()
        st.session_state.node_parents[related_id].add(entity_id)
        added_count += 1

def expand_targeted_entity(entity_id):
    """Expand targeted entity with all its connections and attributes"""
    if entity_id in st.session_state.targeted_expansion:
        return
    
    stored_df = get_stored_entities()
    edges_df = get_relationship_edges()
    
    # Add the targeted entity if not already in graph
    if entity_id not in st.session_state.graph_nodes:
        st.session_state.graph_nodes.add(entity_id)
        entity_data = stored_df[stored_df['entity_id'] == entity_id]
        if not entity_data.empty:
            st.session_state.entity_types[entity_id] = entity_data.iloc[0]['entity_type']
            st.session_state.node_colors[entity_id] = get_entity_type_color(entity_data.iloc[0]['entity_type'])
            st.session_state.node_visibility[entity_id] = True
    
    # Expand all connections for the targeted entity
    connections = get_entity_connections(entity_id, edges_df)
    
    # Add outgoing connections
    for connection in connections['outgoing']:
        target_id = connection['target_id']
        if target_id not in st.session_state.graph_nodes:
            st.session_state.graph_nodes.add(target_id)
            target_data = stored_df[stored_df['entity_id'] == target_id]
            if not target_data.empty:
                st.session_state.entity_types[target_id] = target_data.iloc[0]['entity_type']
                st.session_state.node_colors[target_id] = get_entity_type_color(target_data.iloc[0]['entity_type'])
                st.session_state.node_visibility[target_id] = True
        
        edge = (entity_id, target_id, connection['relationship_type'])
        st.session_state.graph_edges.add(edge)
        st.session_state.edge_visibility[edge] = True
        
        if target_id not in st.session_state.node_parents:
            st.session_state.node_parents[target_id] = set()
        st.session_state.node_parents[target_id].add(entity_id)
    
    # Add incoming connections
    for connection in connections['incoming']:
        source_id = connection['source_id']
        if source_id not in st.session_state.graph_nodes:
            st.session_state.graph_nodes.add(source_id)
            source_data = stored_df[stored_df['entity_id'] == source_id]
            if not source_data.empty:
                st.session_state.entity_types[source_id] = source_data.iloc[0]['entity_type']
                st.session_state.node_colors[source_id] = get_entity_type_color(source_data.iloc[0]['entity_type'])
                st.session_state.node_visibility[source_id] = True
        
        edge = (source_id, entity_id, connection['relationship_type'])
        st.session_state.graph_edges.add(edge)
        st.session_state.edge_visibility[edge] = True
        
        if source_id not in st.session_state.node_parents:
            st.session_state.node_parents[source_id] = set()
        st.session_state.node_parents[source_id].add(entity_id)
    
    st.session_state.targeted_expansion[entity_id] = True

def continuous_expansion(entity_id, depth=2):
    """Recursively expand entities to create continuous expansion"""
    if depth <= 0:
        return
    
    if entity_id not in st.session_state.continuous_expansion:
        st.session_state.continuous_expansion[entity_id] = depth
    
    # Expand the current entity
    expand_targeted_entity(entity_id)
    
    # Get connections for recursive expansion
    edges_df = get_relationship_edges()
    connections = get_entity_connections(entity_id, edges_df)
    
    # Recursively expand connected entities
    all_connected_ids = set()
    for connection in connections['outgoing']:
        all_connected_ids.add(connection['target_id'])
    for connection in connections['incoming']:
        all_connected_ids.add(connection['source_id'])
    
    for connected_id in all_connected_ids:
        if connected_id not in st.session_state.continuous_expansion:
            continuous_expansion(connected_id, depth - 1)

def hierarchical_expansion(entity_id, expansion_level=1):
    """Hierarchical expansion showing attributes and related entities as separate nodes with actual values"""
    if entity_id not in st.session_state.hierarchical_expansion:
        st.session_state.hierarchical_expansion[entity_id] = set()
    
    stored_df = get_stored_entities()
    edges_df = get_relationship_edges()
    
    # Add main entity if not present
    if entity_id not in st.session_state.graph_nodes:
        st.session_state.graph_nodes.add(entity_id)
        entity_data = stored_df[stored_df['entity_id'] == entity_id]
        if not entity_data.empty:
            st.session_state.entity_types[entity_id] = entity_data.iloc[0]['entity_type']
            st.session_state.node_colors[entity_id] = get_entity_type_color(entity_data.iloc[0]['entity_type'])
            st.session_state.node_visibility[entity_id] = True
    
    # Get entity attributes and create attribute nodes
    entity_data = stored_df[stored_df['entity_id'] == entity_id].iloc[0]
    attributes_df = get_entity_attributes(entity_id)
    
    # Parse detailed attributes from JSON
    detailed_attrs = {}
    if entity_data['detailed_attributes']:
        try:
            detailed_attrs = json.loads(entity_data['detailed_attributes'])
        except:
            # Also check attributes field
            if entity_data['attributes']:
                attrs = entity_data['attributes'].split('; ')
                for attr in attrs:
                    if ':' in attr:
                        key, value = attr.split(':', 1)
                        detailed_attrs[key.strip().lower()] = value.strip()
    
    # Define which attributes to show as nodes (for doctors and hospitals)
    important_attributes = []
    entity_type = entity_data['entity_type'].lower()
    
    if entity_type == 'doctor':
        important_attributes = [
            ('specialization', 'Specialization', '#ff7f0e'),
            ('experience', 'Experience', '#ff7f0e'),
            ('success rate', 'Success Rate', '#ff7f0e'),
            ('success_rate', 'Success Rate', '#ff7f0e'),
            ('consultation fees', 'Fees', '#ff7f0e'),
            ('fees', 'Fees', '#ff7f0e'),
            ('availability', 'Availability', '#ff7f0e')
        ]
    elif entity_type in ['hospital', 'clinic']:
        important_attributes = [
            ('location', 'Location', '#1f77b4'),
            ('service hours', 'Service Hours', '#bcbd22'),
            ('service_hours', 'Service Hours', '#bcbd22'),
            ('pharmacy', 'Pharmacy', '#2ca02c'),
            ('ambulance', 'Ambulance', '#d62728'),
            ('nurses', 'Nurses', '#17becf'),
            ('specializations', 'Specializations', '#1f77b4'),
            ('contact', 'Contact', '#1f77b4')
        ]
    elif entity_type == 'location':
        important_attributes = [
            ('full address', 'Address', '#e377c2'),
            ('full_address', 'Address', '#e377c2'),
            ('city', 'City', '#e377c2'),
            ('state', 'State', '#e377c2')
        ]
    elif entity_type == 'service_time':
        important_attributes = [
            ('opening hours', 'Opening', '#bcbd22'),
            ('opening_hours', 'Opening', '#bcbd22'),
            ('closing hours', 'Closing', '#bcbd22'),
            ('closing_hours', 'Closing', '#bcbd22'),
            ('days', 'Days', '#bcbd22'),
            ('emergency services', 'Emergency', '#bcbd22'),
            ('emergency_services', 'Emergency', '#bcbd22')
        ]
    
    # Create attribute nodes for important attributes with actual values
    for attr_key, attr_display_name, attr_color in important_attributes:
        # Check in detailed_attrs first
        attr_value = None
        for key in [attr_key, attr_key.replace(' ', '_'), attr_key.replace('_', ' ')]:
            if key in detailed_attrs:
                attr_value = detailed_attrs[key]
                break
        
        # If not found in detailed_attrs, check in attributes_df
        if not attr_value:
            attr_row = attributes_df[attributes_df['attribute_type'] == attr_key]
            if not attr_row.empty:
                attr_value = attr_row.iloc[0]['attribute_value']
        
        # If still not found, check in raw attributes string
        if not attr_value and entity_data['attributes']:
            attrs = entity_data['attributes'].split('; ')
            for attr in attrs:
                if ':' in attr:
                    key, value = attr.split(':', 1)
                    if key.strip().lower() == attr_key:
                        attr_value = value.strip()
                        break
        
        if attr_value and attr_value.lower() != 'not specified':
            # Create a descriptive node label with the actual value
            node_label = f"{attr_display_name}: {attr_value}"
            
            # Create unique ID for attribute node
            attr_node_id = f"{entity_id}_{attr_key}"
            
            if attr_node_id not in st.session_state.graph_nodes:
                st.session_state.graph_nodes.add(attr_node_id)
                st.session_state.entity_types[attr_node_id] = "attribute"
                st.session_state.node_colors[attr_node_id] = attr_color
                st.session_state.node_visibility[attr_node_id] = True
            
            # Create edge from entity to attribute with a descriptive label
            edge_label = f"has {attr_display_name.lower()}"
            edge = (entity_id, attr_node_id, edge_label)
            if edge not in st.session_state.graph_edges:
                st.session_state.graph_edges.add(edge)
                st.session_state.edge_visibility[edge] = True
            
            if attr_node_id not in st.session_state.node_parents:
                st.session_state.node_parents[attr_node_id] = set()
            st.session_state.node_parents[attr_node_id].add(entity_id)
    
    # Expand to related entities based on entity type
    if expansion_level > 0:
        connections = get_entity_connections(entity_id, edges_df)
        
        # Add related entities
        for connection in connections['outgoing'] + connections['incoming']:
            related_id = connection['target_id'] if 'target_id' in connection else connection['source_id']
            
            if related_id not in st.session_state.graph_nodes:
                st.session_state.graph_nodes.add(related_id)
                related_data = stored_df[stored_df['entity_id'] == related_id]
                if not related_data.empty:
                    st.session_state.entity_types[related_id] = related_data.iloc[0]['entity_type']
                    st.session_state.node_colors[related_id] = get_entity_type_color(related_data.iloc[0]['entity_type'])
                    st.session_state.node_visibility[related_id] = True
            
            # Create appropriate edge
            if 'target_id' in connection:
                edge = (entity_id, related_id, connection['relationship_type'])
            else:
                edge = (related_id, entity_id, connection['relationship_type'])
            
            if edge not in st.session_state.graph_edges:
                st.session_state.graph_edges.add(edge)
                st.session_state.edge_visibility[edge] = True
            
            if related_id not in st.session_state.node_parents:
                st.session_state.node_parents[related_id] = set()
            st.session_state.node_parents[related_id].add(entity_id)
    
    st.session_state.hierarchical_expansion[entity_id].add(expansion_level)

def generate_enhanced_network_graph():
    """Generate enhanced network graph with hierarchical expansion support showing actual values"""
    stored_df = get_stored_entities()
    edges_df = get_relationship_edges()
    
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff" if not st.session_state.graph_options['dark_mode'] else "#1a1a1a",
        font_color="black" if not st.session_state.graph_options['dark_mode'] else "white",
        directed=True,
        notebook=True,
        cdn_resources='remote'
    )
    
    physics_options = {
        "enabled": st.session_state.graph_options['physics_enabled'],
        "barnesHut": {
            "gravitationalConstant": -8000,
            "centralGravity": 0.3,
            "springLength": 200,
            "springConstant": 0.04,
            "damping": 0.09,
            "avoidOverlap": 0.1
        },
        "minVelocity": 0.75,
        "solver": "barnesHut",
        "stabilization": {
            "enabled": True,
            "iterations": 1000,
            "updateInterval": 25
        }
    }
    
    if st.session_state.graph_options['hierarchical_layout']:
        physics_options["solver"] = "hierarchicalRepulsion"
        physics_options["hierarchicalRepulsion"] = {
            "nodeDistance": 150,
            "centralGravity": 0.0,
            "springLength": 200,
            "springConstant": 0.01,
            "damping": 0.09
        }
        net.set_options("""
        var options = {
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "levelSeparation": 150,
                    "nodeSpacing": 100,
                    "treeSpacing": 200,
                    "blockShifting": true,
                    "edgeMinimization": true,
                    "parentCentralization": true,
                    "direction": "UD",
                    "sortMethod": "directed"
                }
            }
        }
        """)
    
    node_options = {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "size": 30,
        "font": {
            "size": st.session_state.graph_options['font_size'],
            "strokeWidth": 2,
            "align": "center",
            "color": "black" if not st.session_state.graph_options['dark_mode'] else "white"
        },
        "scaling": {"min": 10, "max": 50},
        "shadow": {
            "enabled": True,
            "color": "rgba(0,0,0,0.5)",
            "size": 10,
            "x": 5,
            "y": 5
        },
        "shapeProperties": {"useBorderWithImage": True},
        "color": {
            "border": "#2B7CE9",
            "background": "#97C2FC",
            "highlight": {
                "border": "#2B7CE9",
                "background": "#D2E5FF"
            },
            "hover": {
                "border": "#2B7CE9",
                "background": "#D2E5FF"
            }
        }
    }
    
    edge_options = {
        "arrows": {
            "to": {
                "enabled": True,
                "scaleFactor": 0.5,
                "type": "arrow"
            }
        },
        "color": {
            "inherit": True,
            "highlight": "#ff0000",
            "hover": "#ff0000",
            "opacity": 0.8
        },
        "font": {
            "size": st.session_state.graph_options['font_size'] - 2,
            "strokeWidth": 2 if st.session_state.graph_options['show_edge_labels'] else 0,
            "align": "middle",
            "color": "black" if not st.session_state.graph_options['dark_mode'] else "white"
        },
        "smooth": {
            "type": st.session_state.graph_options['edge_smoothness'],
            "roundness": 0.15
        },
        "selectionWidth": 2,
        "shadow": {
            "enabled": True,
            "color": "rgba(0,0,0,0.5)",
            "size": 10,
            "x": 5,
            "y": 5
        },
        "labelHighlightBold": True
    }
    
    net.options = {
        "nodes": node_options,
        "edges": edge_options,
        "physics": physics_options,
        "interaction": {
            "hover": True,
            "multiselect": True,
            "navigationButtons": True,
            "keyboard": True,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": True,
            "hideNodesOnDrag": False
        }
    }
    
    # Add nodes with enhanced tooltips showing all attributes and relationships
    for node_id in st.session_state.graph_nodes:
        if not st.session_state.node_visibility.get(node_id, True):
            continue
        
        # Check if it's an attribute node
        if isinstance(node_id, str) and '_' in node_id and node_id.split('_')[0].isdigit():
            # This is an attribute node
            entity_id_part = node_id.split('_')[0]
            attr_type = '_'.join(node_id.split('_')[1:])
            
            # Try to get the attribute value
            attr_value = ""
            attributes_df = get_entity_attributes(int(entity_id_part))
            for _, row in attributes_df.iterrows():
                if row['attribute_type'] == attr_type:
                    attr_value = row['attribute_value']
                    break
            
            # If not found in attributes table, check detailed_attributes
            if not attr_value:
                entity_data = stored_df[stored_df['entity_id'] == int(entity_id_part)]
                if not entity_data.empty:
                    detailed_attrs = safe_json_loads(entity_data.iloc[0]['detailed_attributes'])
                    # Try different key formats
                    for key in [attr_type, attr_type.replace('_', ' ')]:
                        if key in detailed_attrs:
                            attr_value = detailed_attrs[key]
                            break
            
            # Create a clean display name
            display_name = attr_type.replace('_', ' ').title()
            if attr_value:
                display_name = f"{display_name}: {attr_value}"
            
            node_tooltip = f"""<div style="max-width: 300px; padding: 10px; background-color: {'#ffffff' if not st.session_state.graph_options['dark_mode'] else '#2d2d2d'}; 
                color: {'black' if not st.session_state.graph_options['dark_mode'] else 'white'}; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.3); border-left: 4px solid {st.session_state.node_colors.get(node_id, '#8c564b')};">
                <h4 style="margin: 0 0 10px 0; padding: 0; color: {st.session_state.node_colors.get(node_id, '#8c564b')}; border-bottom: 1px solid #eee; padding-bottom: 5px;">{display_name}</h4>
                <div style="font-size: 12px; line-height: 1.4;">
                    <p style="margin: 5px 0;"><b>Type:</b> Attribute</p>
                    <p style="margin: 5px 0;"><b>Attribute:</b> {attr_type.replace('_', ' ').title()}</p>
                    {f'<p style="margin: 5px 0;"><b>Value:</b> {attr_value}</p>' if attr_value else ''}
                </div>
            </div>"""
            
            net.add_node(
                node_id, 
                label=display_name if st.session_state.graph_options['show_node_labels'] else "", 
                title=node_tooltip, 
                color=st.session_state.node_colors.get(node_id, "#8c564b"), 
                size=20,  # Slightly larger for attribute nodes
                level=2, 
                shape="box", 
                borderWidth=2,
                font={'size': st.session_state.graph_options['font_size'] - 1}
            )
        else:
            # Regular entity node
            node_data = stored_df[stored_df['entity_id'] == node_id]
            if not node_data.empty:
                node_row = node_data.iloc[0]
                node_name = node_row['entity_name']
                node_type = node_row['entity_type']
                node_weight = node_row['weight']
                
                level = 0
                if node_id in st.session_state.node_parents:
                    level = max([st.session_state.hierarchy_level.get(p, 0) for p in st.session_state.node_parents[node_id]]) + 1
                
                st.session_state.hierarchy_level[node_id] = level
                node_color = st.session_state.node_colors.get(node_id, get_entity_type_color(node_type))
                node_size = 20 + (node_weight * st.session_state.graph_options['node_size_multiplier'])
                
                # Enhanced tooltip with all attributes and relationships
                detailed_attrs = ""
                if node_row['detailed_attributes'] and st.session_state.graph_options['show_attribute_details']:
                    try:
                        attrs = safe_json_loads(node_row['detailed_attributes'])
                        if attrs:
                            detailed_attrs = "<br><b>Attributes:</b><ul>" + "".join([f"<li><b>{k.replace('_', ' ').title()}:</b> {v}</li>" for k, v in attrs.items() if v and v != 'Not specified']) + "</ul>"
                    except:
                        pass
                
                # Show all relationships
                connections = []
                for edge in st.session_state.graph_edges:
                    if edge[0] == node_id or edge[1] == node_id:
                        if edge[0] == node_id:
                            target_id = edge[1]
                            if isinstance(target_id, str) and '_' in target_id:
                                # Attribute connection
                                attr_type = '_'.join(target_id.split('_')[1:])
                                # Try to get attribute value for display
                                attr_value = ""
                                if node_row['detailed_attributes']:
                                    try:
                                        attrs = safe_json_loads(node_row['detailed_attributes'])
                                        for key in [attr_type, attr_type.replace('_', ' ')]:
                                            if key in attrs:
                                                attr_value = attrs[key]
                                                break
                                    except:
                                        pass
                                if attr_value:
                                    connections.append(f"→ {attr_type.replace('_', ' ').title()}: {attr_value}")
                                else:
                                    connections.append(f"→ {attr_type.replace('_', ' ').title()}")
                            else:
                                target_data = stored_df[stored_df['entity_id'] == target_id]
                                if not target_data.empty:
                                    connections.append(f"→ {target_data.iloc[0]['entity_name']} ({edge[2]})")
                        else:
                            source_id = edge[0]
                            if isinstance(source_id, str) and '_' in source_id:
                                # Attribute connection (shouldn't happen for incoming)
                                pass
                            else:
                                source_data = stored_df[stored_df['entity_id'] == source_id]
                                if not source_data.empty:
                                    connections.append(f"← {source_data.iloc[0]['entity_name']} ({edge[2]})")
                
                # Enhanced tooltip content
                node_tooltip = f"""<div style="max-width: 400px; padding: 10px; background-color: {'#ffffff' if not st.session_state.graph_options['dark_mode'] else '#2d2d2d'}; 
                    color: {'black' if not st.session_state.graph_options['dark_mode'] else 'white'}; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.3); border-left: 4px solid {node_color};">
                    <h4 style="margin: 0 0 10px 0; padding: 0; color: {node_color}; border-bottom: 1px solid #eee; padding-bottom: 5px;">{node_name}</h4>
                    <div style="font-size: 12px; line-height: 1.4;">
                        <p style="margin: 5px 0;"><b>Type:</b> {node_type} | <b>Weight:</b> {node_weight:.2f} | <b>Level:</b> {level}</p>
                        <p style="margin: 5px 0;"><b>Attributes:</b> {len(node_row['attributes'].split(';')) if node_row['attributes'] else 0} | 
                        <b>Connections:</b> {len([e for e in st.session_state.graph_edges if e[0] == node_id or e[1] == node_id])}</p>
                        {detailed_attrs}
                        <br><b>All Connections:</b><ul style="max-height: 200px; overflow-y: auto; margin: 5px 0; padding-left: 15px;">{"".join([f"<li style='margin: 2px 0;'>{conn}</li>" for conn in connections])}</ul>
                    </div>
                </div>"""
                
                net.add_node(
                    node_id,
                    label=node_name if st.session_state.graph_options['show_node_labels'] else "",
                    title=node_tooltip,
                    color=node_color,
                    size=node_size,
                    level=level,
                    shape=st.session_state.graph_options['node_shape'],
                    borderWidth=3,
                    mass=1 + (node_weight * 0.5),
                    group=node_type if st.session_state.graph_options['cluster_nodes'] else None,
                    font={'size': st.session_state.graph_options['font_size']}
                )
    
    # Add edges
    for edge in st.session_state.graph_edges:
        if not st.session_state.edge_visibility.get(edge, True):
            continue
        
        source_id, target_id, relationship = edge
        
        # Handle attribute edges differently
        if isinstance(target_id, str) and '_' in target_id:
            # This is an attribute edge
            # Extract attribute type
            attr_type = '_'.join(target_id.split('_')[1:])
            
            # Try to get the actual value for the edge label
            attr_value = ""
            if isinstance(source_id, int):
                attributes_df = get_entity_attributes(source_id)
                for _, row in attributes_df.iterrows():
                    if row['attribute_type'] == attr_type:
                        attr_value = row['attribute_value']
                        break
                
                # If not found in attributes table, check detailed_attributes
                if not attr_value:
                    source_data = stored_df[stored_df['entity_id'] == source_id]
                    if not source_data.empty:
                        detailed_attrs = safe_json_loads(source_data.iloc[0]['detailed_attributes'])
                        # Try different key formats
                        for key in [attr_type, attr_type.replace('_', ' ')]:
                            if key in detailed_attrs:
                                attr_value = detailed_attrs[key]
                                break
            
            # Create a more descriptive edge label
            if attr_value and st.session_state.graph_options['show_attribute_values']:
                edge_label = f"{relationship}: {attr_value}"
            else:
                edge_label = relationship
            
            title = f"{relationship}: {attr_value}" if attr_value else relationship
            width = 1.5  # Thicker for attribute edges
        else:
            edge_data = edges_df[(edges_df['source_id'] == source_id) & (edges_df['target_id'] == target_id)]
            if not edge_data.empty:
                edge_row = edge_data.iloc[0]
                title = edge_row['full_description']
                weight = edge_row['weight']
                edge_label = edge_row['relationship_type']
            else:
                title = relationship
                weight = 1.0
                edge_label = relationship
            width = 1 + (weight * st.session_state.graph_options['edge_width_multiplier'])
        
        net.add_edge(
            source_id,
            target_id,
            title=title,
            label=edge_label if st.session_state.graph_options['show_edge_labels'] else "",
            width=width,
            smooth=True,
            arrowStrikethrough=False,
            hidden=False,
            selectionWidth=1,
            color={'inherit': 'both'},
            font={'strokeWidth': 3, 'size': st.session_state.graph_options['font_size'] - 1}
        )
    
    graph_path = f"temp_graph_{uuid.uuid4()}.html"
    net.save_graph(graph_path)
    
    # Add custom JavaScript for enhanced interactivity
    with open(graph_path, 'r+', encoding='utf-8') as f:
        content = f.read()
        f.seek(0, 0)
        custom_js = """
        <script>
            document.addEventListener("DOMContentLoaded", function() {
                setTimeout(function() {
                    const container = document.getElementsByClassName("vis-network")[0];
                    network.fit(); 
                    network.stabilize(1000); 
                    network.moveTo({
                        position: {x: 0, y: 0},
                        scale: 0.9,
                        offset: {x: 0, y: 0},
                        animation: {
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                    
                    // Enhanced double-click for hierarchical expansion
                    network.on("doubleClick", function(params) {
                        if (params.nodes.length > 0) {
                            const nodeId = params.nodes[0];
                            const nodeData = network.body.data.nodes.get(nodeId);
                            if (nodeData) {
                                // Trigger hierarchical expansion
                                console.log("Double-clicked node for hierarchical expansion:", nodeData.label, nodeId);
                                // This would need to be connected to Streamlit backend
                            }
                        }
                    });
                    
                    network.on("hoverNode", function(params) {
                        network.canvas.body.container.style.cursor = 'pointer';
                    });
                    
                    network.on("blurNode", function(params) {
                        network.canvas.body.container.style.cursor = 'default';
                    });
                    
                }, 500);
            });
        </script>
        """
        content = content.replace('</body>', custom_js + '</body>')
        f.write(content)
        f.truncate()
    
    return graph_path

def toggle_node_expansion(entity_id):
    if entity_id in st.session_state.expanded_nodes:
        st.session_state.expanded_nodes.remove(entity_id)
        nodes_to_remove = set()
        edges_to_remove = set()
        
        for node_id in st.session_state.graph_nodes:
            if node_id != entity_id and node_id in st.session_state.node_parents:
                if len(st.session_state.node_parents[node_id]) == 1 and entity_id in st.session_state.node_parents[node_id]:
                    nodes_to_remove.add(node_id)
        
        for edge in st.session_state.graph_edges.copy():
            source_id, target_id, _ = edge
            if source_id in nodes_to_remove or target_id in nodes_to_remove:
                edges_to_remove.add(edge)
        
        st.session_state.graph_nodes -= nodes_to_remove
        st.session_state.graph_edges -= edges_to_remove
    else:
        st.session_state.expanded_nodes.add(entity_id)
        expand_targeted_entity(entity_id)

def expand_node(entity_id):
    stored_df = get_stored_entities()
    edges_df = get_relationship_edges()
    expansion_limit = st.session_state.graph_options.get('expansion_limit', 5)
    added_count = 0
    
    outgoing = edges_df[edges_df['source_id'] == entity_id]
    for _, edge in outgoing.iterrows():
        if added_count >= expansion_limit:
            break
        
        target_id = edge['target_id']
        if target_id not in st.session_state.graph_nodes:
            st.session_state.graph_nodes.add(target_id)
            target_data = stored_df[stored_df['entity_id'] == target_id]
            if not target_data.empty:
                st.session_state.entity_types[target_id] = target_data.iloc[0]['entity_type']
                st.session_state.node_colors[target_id] = get_entity_type_color(target_data.iloc[0]['entity_type'])
                st.session_state.node_visibility[target_id] = True
        
        edge_tuple = (entity_id, target_id, edge['relationship_type'])
        st.session_state.graph_edges.add(edge_tuple)
        st.session_state.edge_visibility[edge_tuple] = True
        
        if target_id not in st.session_state.node_parents:
            st.session_state.node_parents[target_id] = set()
        st.session_state.node_parents[target_id].add(entity_id)
        added_count += 1
    
    incoming = edges_df[edges_df['target_id'] == entity_id]
    for _, edge in incoming.iterrows():
        if added_count >= expansion_limit * 2:
            break
        
        source_id = edge['source_id']
        if source_id not in st.session_state.graph_nodes:
            st.session_state.graph_nodes.add(source_id)
            source_data = stored_df[stored_df['entity_id'] == source_id]
            if not source_data.empty:
                st.session_state.entity_types[source_id] = source_data.iloc[0]['entity_type']
                st.session_state.node_colors[source_id] = get_entity_type_color(source_data.iloc[0]['entity_type'])
                st.session_state.node_visibility[source_id] = True
        
        edge_tuple = (source_id, entity_id, edge['relationship_type'])
        st.session_state.graph_edges.add(edge_tuple)
        st.session_state.edge_visibility[edge_tuple] = True
        
        if source_id not in st.session_state.node_parents:
            st.session_state.node_parents[source_id] = set()
        st.session_state.node_parents[source_id].add(entity_id)
        added_count += 1

def reset_graph():
    st.session_state.graph_nodes = set()
    st.session_state.graph_edges = set()
    st.session_state.expanded_nodes = set()
    st.session_state.node_colors = {}
    st.session_state.entity_types = {}
    st.session_state.hierarchy_level = {}
    st.session_state.node_parents = {}
    st.session_state.expanded_by_type = {}
    st.session_state.selected_node = None
    st.session_state.relationship_filter = None
    st.session_state.node_visibility = {}
    st.session_state.edge_visibility = {}
    st.session_state.targeted_expansion = {}
    st.session_state.continuous_expansion = {}
    st.session_state.hierarchical_expansion = {}

def visualize_knowledge_graph():
    stored_df = get_stored_entities()
    edges_df = get_relationship_edges()
    initialize_graph_state()
    
    st.markdown("""
    <style>
        .graph-section {background-color: #f8f9fa;padding: 1.5rem;border-radius: 8px;margin-bottom: 1.5rem;box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
        .graph-controls {background-color: white;padding: 1rem;border-radius: 8px;margin-bottom: 1rem;box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
        .entity-list {background-color: white;padding: 1rem;border-radius: 8px;box-shadow: 0 1px 3px rgba(0,0,0,0.1);max-height: 600px;overflow-y: auto;}
        .node-visibility {background-color: white;padding: 1rem;border-radius: 8px;box-shadow: 0 1px 3px rgba(0,0,0,0.1);max-height: 600px;overflow-y: auto;}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="graph-section">', unsafe_allow_html=True)
    st.subheader("🌐 Advanced Knowledge Graph Visualization")
    
    with st.expander("⚙️ Graph Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.graph_options['node_size_multiplier'] = st.slider(
                "Node Size Multiplier", 1, 10, 5,
                help="Adjust the size of nodes based on their importance"
            )
            st.session_state.graph_options['edge_width_multiplier'] = st.slider(
                "Edge Width Multiplier", 0.1, 2.0, 0.5, 0.1,
                help="Adjust the width of edges based on relationship strength"
            )
            st.session_state.graph_options['font_size'] = st.slider(
                "Font Size", 8, 20, 12,
                help="Adjust the font size for node and edge labels"
            )
        with col2:
            st.session_state.graph_options['physics_enabled'] = st.checkbox(
                "Enable Physics", True,
                help="Enable physics simulation for dynamic graph layout"
            )
            st.session_state.graph_options['hierarchical_layout'] = st.checkbox(
                "Hierarchical Layout", False,
                help="Use hierarchical layout for better visualization of relationships"
            )
            st.session_state.graph_options['cluster_nodes'] = st.checkbox(
                "Cluster by Type", True,
                help="Group nodes by their entity type"
            )
            st.session_state.graph_options['dark_mode'] = st.checkbox(
                "Dark Mode", False,
                help="Toggle dark mode for the graph"
            )
        with col3:
            st.session_state.graph_options['show_edge_labels'] = st.checkbox(
                "Show Edge Labels", True,
                help="Display relationship labels on edges"
            )
            st.session_state.graph_options['show_node_labels'] = st.checkbox(
                "Show Node Labels", True,
                help="Display entity names on nodes"
            )
            st.session_state.graph_options['show_attribute_details'] = st.checkbox(
                "Show Attribute Details", True,
                help="Show detailed attributes in node tooltips"
            )
            st.session_state.graph_options['show_attribute_values'] = st.checkbox(
                "Show Attribute Values", True,
                help="Show actual attribute values on edges"
            )
            st.session_state.graph_options['expansion_limit'] = st.slider(
                "Expansion Limit", 1, 10, 5,
                help="Number of connections to show when expanding a node"
            )
    
    st.markdown('<div class="graph-controls">', unsafe_allow_html=True)
    st.write("### 🔍 Filter Options")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        entity_type_filter = st.multiselect(
            "Filter by Entity Type",
            ["All"] + sorted(stored_df['entity_type'].unique().tolist()),
            default=["All"],
            key="entity_type_filter"
        )
    
    with filter_col2:
        min_weight = st.slider(
            "Minimum Entity Weight",
            min_value=1.0,
            max_value=float(stored_df['weight'].max()) if not stored_df.empty else 10.0,
            value=1.0,
            step=0.5,
            key="min_weight"
        )
    
    with filter_col3:
        search_term = st.text_input(
            "Search Entities",
            key="graph_search",
            placeholder="Search by name, attributes, or relationships"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    filtered_df = stored_df.copy()
    if "All" not in entity_type_filter:
        filtered_df = filtered_df[filtered_df['entity_type'].isin(entity_type_filter)]
    filtered_df = filtered_df[filtered_df['weight'] >= min_weight]
    
    if search_term:
        filtered_df = search_entities(search_term, filtered_df)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown('<div class="entity-list">', unsafe_allow_html=True)
        st.write("### 📌 Entities")
        st.write("Click to add/remove entities from the graph:")
        
        if not filtered_df.empty:
            # Entity Details View Toggle
            show_details = st.checkbox("Show Entity Details", value=False, help="Show detailed information for each entity")
            
            entity_types = [et for et in sorted(filtered_df['entity_type'].unique()) 
                          if et.lower() in ['hospital', 'clinic', 'doctor', 'pharmacy', 'ambulance_service', 'person', 'organization', 'location', 'nurse', 'service_time']]
            
            for entity_type in entity_types:
                type_df = filtered_df[filtered_df['entity_type'] == entity_type]
                type_color = get_entity_type_color(entity_type)
                
                with st.expander(f"📌 {entity_type.capitalize()} ({len(type_df)})", expanded=True):
                    for _, row in type_df.iterrows():
                        entity_id = row['entity_id']
                        entity_name = row['entity_name']
                        entity_weight = row['weight']
                        
                        if show_details:
                            # Show detailed entity information
                            with st.container():
                                st.markdown(f"""
                                <div style="background-color: {type_color}20; padding: 0.75rem; border-radius: 5px; margin-bottom: 0.5rem; border-left: 3px solid {type_color};">
                                    <strong>{entity_name}</strong><br>
                                    <small>Weight: {entity_weight:.1f} | Type: {entity_type}</small>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show key attributes
                                detailed_attrs = safe_json_loads(row.get('detailed_attributes', '{}'))
                                if detailed_attrs:
                                    for key, value in list(detailed_attrs.items())[:2]:  # Show first 2 attributes
                                        if value and value != 'Not specified':
                                            st.markdown(f"<small><strong>{key.replace('_', ' ').title()}:</strong> {value}</small>", unsafe_allow_html=True)
                                
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    button_color = "primary" if entity_id in st.session_state.graph_nodes else "secondary"
                                    if st.button("Add/Remove", key=f"add_{entity_id}", use_container_width=True):
                                        if entity_id in st.session_state.graph_nodes:
                                            st.session_state.graph_nodes.remove(entity_id)
                                            st.session_state.graph_edges = {edge for edge in st.session_state.graph_edges if edge[0] != entity_id and edge[1] != entity_id}
                                            if entity_id in st.session_state.expanded_nodes:
                                                st.session_state.expanded_nodes.remove(entity_id)
                                        else:
                                            st.session_state.graph_nodes.add(entity_id)
                                            st.session_state.entity_types[entity_id] = entity_type
                                            st.session_state.node_colors[entity_id] = type_color
                                            st.session_state.node_visibility[entity_id] = True
                                        st.rerun()
                                with col_b:
                                    if st.button("View Details", key=f"view_{entity_id}", use_container_width=True):
                                        st.session_state.selected_node = entity_id
                                        st.rerun()
                        else:
                            # Compact view
                            col_a, col_b, col_c = st.columns([3, 1, 1])
                            with col_a:
                                button_color = "primary" if entity_id in st.session_state.graph_nodes else "secondary"
                                if st.button(
                                    f"{entity_name} ({entity_weight:.1f})",
                                    key=f"entity_{entity_id}",
                                    type=button_color,
                                    use_container_width=True
                                ):
                                    if entity_id in st.session_state.graph_nodes:
                                        st.session_state.graph_nodes.remove(entity_id)
                                        st.session_state.graph_edges = {edge for edge in st.session_state.graph_edges if edge[0] != entity_id and edge[1] != entity_id}
                                        if entity_id in st.session_state.expanded_nodes:
                                            st.session_state.expanded_nodes.remove(entity_id)
                                    else:
                                        st.session_state.graph_nodes.add(entity_id)
                                        st.session_state.entity_types[entity_id] = entity_type
                                        st.session_state.node_colors[entity_id] = type_color
                                        st.session_state.node_visibility[entity_id] = True
                                    st.rerun()
                            with col_b:
                                if st.button("🔍", key=f"expand_{entity_id}", help=f"Expand {entity_name} with all connections"):
                                    expand_targeted_entity(entity_id)
                                    st.rerun()
                            with col_c:
                                if st.button("📋", key=f"details_{entity_id}", help=f"View details for {entity_name}"):
                                    st.session_state.selected_node = entity_id
                                    st.rerun()
        
        if st.button("🔄 Reset Graph", type="primary", use_container_width=True):
            reset_graph()
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="node-visibility">', unsafe_allow_html=True)
        st.write("### 👁️ Node Visibility")
        
        if st.session_state.graph_nodes:
            for node_id in sorted(st.session_state.graph_nodes, key=str):
                if isinstance(node_id, str) and '_' in node_id:
                    # Attribute node
                    entity_id_part = node_id.split('_')[0]
                    attr_type = '_'.join(node_id.split('_')[1:])
                    node_name = f"Attribute: {attr_type.replace('_', ' ').title()}"
                else:
                    # Regular entity node
                    node_data = stored_df[stored_df['entity_id'] == node_id]
                    if not node_data.empty:
                        node_name = node_data.iloc[0]['entity_name']
                    else:
                        node_name = f"Node {node_id}"
                
                current_visibility = st.session_state.node_visibility.get(node_id, True)
                new_visibility = st.checkbox(f"Show {node_name}", value=current_visibility, key=f"node_vis_{node_id}")
                
                if new_visibility != current_visibility:
                    st.session_state.node_visibility[node_id] = new_visibility
                    st.rerun()
        else:
            st.info("No nodes in graph yet")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
    
    # Show detailed entity view if selected
    if st.session_state.selected_node:
        st.write("### 📋 Entity Details")
        display_entity_details(st.session_state.selected_node, stored_df)
        if st.button("Close Details"):
            st.session_state.selected_node = None
            st.rerun()
    
    st.write("### 🌐 Graph Visualization")
    graph_container = st.container()
    
    if st.session_state.graph_nodes:
        graph_path = generate_enhanced_network_graph()
        
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_html = f.read()
        
        graph_html = graph_html.replace(
            '<div id="mynetwork"></div>',
            '<div id="mynetwork" style="width: 100%; height: 800px; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"></div>'
        )
        
        with graph_container:
            components.html(graph_html, height=850, scrolling=False)
        
        try:
            os.remove(graph_path)
        except:
            pass
        
        # Enhanced Node Exploration Section
        st.write("### 🔍 Advanced Node Exploration")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Basic Expansion", "Type-Based Expansion", "Continuous Expansion", "Hierarchical Expansion"])
        
        with tab1:
            st.write("Expand individual nodes to see immediate connections:")
            for entity_id in sorted(st.session_state.graph_nodes, key=str):
                if not isinstance(entity_id, str):  # Only show regular entities, not attributes
                    entity_data = stored_df[stored_df['entity_id'] == entity_id]
                    if not entity_data.empty:
                        entity_name = entity_data.iloc[0]['entity_name']
                        entity_type = entity_data.iloc[0]['entity_type']
                        button_color = "primary" if entity_id in st.session_state.expanded_nodes else "secondary"
                        if st.button(f"🔍 Expand {entity_name} ({entity_type})", key=f"expand_basic_{entity_id}", type=button_color, use_container_width=True):
                            toggle_node_expansion(entity_id)
                            st.rerun()
        
        with tab2:
            st.write("Expand nodes by specific connection types:")
            for entity_id in sorted(st.session_state.graph_nodes, key=str):
                if not isinstance(entity_id, str):  # Only show regular entities, not attributes
                    entity_data = stored_df[stored_df['entity_id'] == entity_id]
                    if not entity_data.empty:
                        entity_name = entity_data.iloc[0]['entity_name']
                        connected_types = get_entity_types_from_connections(entity_id)
                        if connected_types:
                            st.write(f"**{entity_name}** connected to:")
                            for entity_type in connected_types:
                                if entity_type.lower() in ['hospital', 'clinic', 'doctor', 'pharmacy', 'ambulance_service', 'person', 'organization', 'location', 'nurse', 'service_time']:
                                    if entity_id not in st.session_state.expanded_by_type:
                                        st.session_state.expanded_by_type[entity_id] = set()
                                    is_expanded = entity_type in st.session_state.expanded_by_type.get(entity_id, set())
                                    button_color = "primary" if is_expanded else "secondary"
                                    if st.button(f"🔗 Show {entity_type} connections", key=f"type_{entity_id}_{entity_type}", type=button_color, use_container_width=True):
                                        expand_node_by_type(entity_id, entity_type)
                                        st.rerun()
        
        with tab3:
            st.write("Continuous expansion with configurable depth:")
            expansion_depth_continuous = st.slider(
                "Expansion Depth", 1, 5, 2, key="continuous_depth",
                help="Number of levels to expand recursively"
            )
            
            for entity_id in sorted(st.session_state.graph_nodes, key=str):
                if not isinstance(entity_id, str):  # Only show regular entities, not attributes
                    entity_data = stored_df[stored_df['entity_id'] == entity_id]
                    if not entity_data.empty:
                        entity_name = entity_data.iloc[0]['entity_name']
                        if st.button(f"🔄 Continuously Expand {entity_name}", key=f"continuous_{entity_id}", type="primary", use_container_width=True):
                            continuous_expansion(entity_id, expansion_depth_continuous)
                            st.rerun()
        
        with tab4:
            st.write("### 🏗️ Hierarchical Expansion with Attribute Values")
            st.info("This expansion shows actual attribute values like 'Success: 70%', 'Experience: 15 years', 'Fees: $200' as separate nodes in the graph.")
            
            hierarchical_depth = st.slider(
                "Hierarchical Depth", 1, 3, 1, key="hierarchical_tab_depth",
                help="Depth for hierarchical expansion"
            )
            
            # Show current entities that can be hierarchically expanded
            entities_to_expand = []
            for entity_id in sorted(st.session_state.graph_nodes, key=str):
                if not isinstance(entity_id, str):  # Only show regular entities, not attributes
                    entity_data = stored_df[stored_df['entity_id'] == entity_id]
                    if not entity_data.empty:
                        entities_to_expand.append((entity_id, entity_data.iloc[0]['entity_name'], entity_data.iloc[0]['entity_type']))
            
            if entities_to_expand:
                # Group by entity type for better organization
                entity_type_groups = {}
                for entity_id, entity_name, entity_type in entities_to_expand:
                    if entity_type not in entity_type_groups:
                        entity_type_groups[entity_type] = []
                    entity_type_groups[entity_type].append((entity_id, entity_name))
                
                for entity_type, entities in entity_type_groups.items():
                    st.write(f"**{entity_type.capitalize()} Entities:**")
                    for entity_id, entity_name in entities:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{entity_name}**")
                        with col2:
                            if st.button(f"🏗️ Expand", key=f"hierarchical_tab_{entity_id}", use_container_width=True):
                                hierarchical_expansion(entity_id, hierarchical_depth)
                                st.success(f"Hierarchically expanded {entity_name} - showing attribute values in graph!")
                                st.rerun()
            else:
                st.info("Add entities to the graph first to use hierarchical expansion.")
    
    else:
        st.info("ℹ️ Add entities to the graph from the left sidebar to start visualizing.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def analyze_entity_relationships():
    st.subheader("🔗 Entity Relationship Analysis")
    stored_df = get_stored_entities()
    edges_df = get_relationship_edges()
    
    st.markdown("""
    <style>
        .analysis-section {background-color: #f8f9fa;padding: 1.5rem;border-radius: 8px;margin-bottom: 1.5rem;box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
        .analysis-controls {background-color: white;padding: 1rem;border-radius: 8px;margin-bottom: 1rem;box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    
    with st.markdown('<div class="analysis-controls">', unsafe_allow_html=True):
        col1, col2 = st.columns(2)
        with col1:
            entity_type_filter = st.multiselect(
                "Filter by Entity Type",
                ["All"] + sorted(stored_df['entity_type'].unique().tolist()),
                default=["All"],
                key="entity_type_filter_analysis"
            )
        with col2:
            min_weight = st.slider(
                "Minimum Entity Weight",
                min_value=1.0,
                max_value=float(stored_df['weight'].max()) if not stored_df.empty else 10.0,
                value=1.0,
                step=0.5,
                key="min_weight_analysis"
            )
    st.markdown('</div>', unsafe_allow_html=True)
    
    filtered_df = stored_df.copy()
    if "All" not in entity_type_filter:
        filtered_df = filtered_df[filtered_df['entity_type'].isin(entity_type_filter)]
    filtered_df = filtered_df[filtered_df['weight'] >= min_weight]
    
    with st.markdown('<div class="analysis-controls">', unsafe_allow_html=True):
        col1, col2 = st.columns(2)
        with col1:
            entity1_id = st.selectbox(
                "Select First Entity",
                options=filtered_df['entity_id'].tolist(),
                format_func=lambda x: filtered_df[filtered_df['entity_id'] == x].iloc[0]['entity_name'],
                key="entity1_select"
            )
        with col2:
            entity2_id = st.selectbox(
                "Select Second Entity",
                options=filtered_df['entity_id'].tolist(),
                format_func=lambda x: filtered_df[filtered_df['entity_id'] == x].iloc[0]['entity_name'],
                index=1 if len(filtered_df) > 1 else 0,
                key="entity2_select"
            )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if entity1_id and entity2_id:
        entity1_name = filtered_df[filtered_df['entity_id'] == entity1_id].iloc[0]['entity_name']
        entity2_name = filtered_df[filtered_df['entity_id'] == entity2_id].iloc[0]['entity_name']
        
        st.write(f"### Analyzing relationship between: **{entity1_name}** and **{entity2_name}**")
        
        # Show entity details side by side
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{entity1_name}**")
            display_entity_details(entity1_id, stored_df)
        
        with col2:
            st.write(f"**{entity2_name}**")
            display_entity_details(entity2_id, stored_df)
        
        direct_relationship = edges_df[(edges_df['source_id'] == entity1_id) & (edges_df['target_id'] == entity2_id)]
        reverse_relationship = edges_df[(edges_df['source_id'] == entity2_id) & (edges_df['target_id'] == entity1_id)]
        
        if not direct_relationship.empty or not reverse_relationship.empty:
            st.write("#### Direct Relationships")
            if not direct_relationship.empty:
                for _, row in direct_relationship.iterrows():
                    st.markdown(f"""<div style="background-color: #e8f4fc; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                        <b>{entity1_name}</b> → <b>{entity2_name}</b>: {row['full_description']}</div>""", unsafe_allow_html=True)
            if not reverse_relationship.empty:
                for _, row in reverse_relationship.iterrows():
                    st.markdown(f"""<div style="background-color: #e8f4fc; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                        <b>{entity2_name}</b> → <b>{entity1_name}</b>: {row['full_description']}</div>""", unsafe_allow_html=True)
        
        st.write("#### Connection Paths")
        paths = find_all_paths(edges_df, entity1_id, entity2_id)
        if paths:
            st.write(f"Found {len(paths)} path(s) between these entities:")
            for i, path in enumerate(paths, 1):
                path_str = []
                for j in range(len(path) - 1):
                    source_id = path[j]
                    target_id = path[j + 1]
                    source_name = filtered_df[filtered_df['entity_id'] == source_id].iloc[0]['entity_name']
                    target_name = filtered_df[filtered_df['entity_id'] == target_id].iloc[0]['entity_name']
                    rel_edge = edges_df[(edges_df['source_id'] == source_id) & (edges_df['target_id'] == target_id)]
                    if not rel_edge.empty:
                        relationship = rel_edge.iloc[0]['relationship_type']
                    else:
                        relationship = "related to"
                    path_str.append(f"**{source_name}** → [{relationship}] → **{target_name}**")
                st.markdown(f"""<div style="background-color: #f0f7ff; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    <b>Path {i}:</b> {' → '.join(path_str)}</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="background-color: #fff3cd; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                No connection paths found between <b>{entity1_name}</b> and <b>{entity2_name}</b>.</div>""", unsafe_allow_html=True)
        
        st.write("#### Common Connections")
        entity1_outgoing = set(edges_df[edges_df['source_id'] == entity1_id]['target_id'])
        entity2_outgoing = set(edges_df[edges_df['source_id'] == entity2_id]['target_id'])
        common_outgoing = entity1_outgoing.intersection(entity2_outgoing)
        
        if common_outgoing:
            st.write(f"Both **{entity1_name}** and **{entity2_name}** connect to:")
            for common_id in common_outgoing:
                common_name = filtered_df[filtered_df['entity_id'] == common_id].iloc[0]['entity_name']
                rel1 = edges_df[(edges_df['source_id'] == entity1_id) & (edges_df['target_id'] == common_id)].iloc[0]['relationship_type']
                rel2 = edges_df[(edges_df['source_id'] == entity2_id) & (edges_df['target_id'] == common_id)].iloc[0]['relationship_type']
                st.markdown(f"""<div style="background-color: #e8f5e9; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    <b>{common_name}</b> ({entity1_name} {rel1}; {entity2_name} {rel2})</div>""", unsafe_allow_html=True)
        
        entity1_incoming = set(edges_df[edges_df['target_id'] == entity1_id]['source_id'])
        entity2_incoming = set(edges_df[edges_df['target_id'] == entity2_id]['source_id'])
        common_incoming = entity1_incoming.intersection(entity2_incoming)
        
        if common_incoming:
            st.write(f"Both **{entity1_name}** and **{entity2_name}** are connected from:")
            for common_id in common_incoming:
                common_name = filtered_df[filtered_df['entity_id'] == common_id].iloc[0]['entity_name']
                rel1 = edges_df[(edges_df['source_id'] == common_id) & (edges_df['target_id'] == entity1_id)].iloc[0]['relationship_type']
                rel2 = edges_df[(edges_df['source_id'] == common_id) & (edges_df['target_id'] == entity2_id)].iloc[0]['relationship_type']
                st.markdown(f"""<div style="background-color: #e8f5e9; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    <b>{common_name}</b> ({common_name} {rel1} {entity1_name}; {common_name} {rel2} {entity2_name})</div>""", unsafe_allow_html=True)
        
        if not common_outgoing and not common_incoming:
            st.markdown(f"""<div style="background-color: #fff3cd; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                No common connections found between <b>{entity1_name}</b> and <b>{entity2_name}</b>.</div>""", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Healthcare Knowledge Graph Explorer",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/healthcare-kg',
            'Report a bug': "https://github.com/yourusername/healthcare-kg/issues",
            'About': "# AI-Powered Healthcare Knowledge Graph Explorer\n\nExtract and visualize healthcare entities using AI"
        }
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
        .stApp {background-color: #f8f9fa;}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {color: #2c3e50;}
        .stButton>button {border-radius: 8px;padding: 0.5rem 1rem;font-weight: 500;transition: all 0.2s ease;background-color: #6e8efb;color: white;border: none;}
        .stButton>button:hover {transform: translateY(-2px);box-shadow: 0 2px 6px rgba(110, 142, 251, 0.4);background-color: #5a7de3;}
        .stButton>button:focus {box-shadow: 0 0 0 0.2rem rgba(110, 142, 251, 0.25);}
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {border-radius: 8px;border: 1px solid #ced4da;padding: 0.5rem;}
        .stSelectbox>div>div>select {border-radius: 8px;border: 1px solid #ced4da;padding: 0.5rem;}
        .stTabs [data-baseweb="tab-list"] {gap: 8px;padding: 0 1rem;}
        .stTabs [data-baseweb="tab"] {padding: 0.75rem 1.5rem;border-radius: 8px 8px 0 0;font-weight: 500;transition: all 0.2s ease;background-color: #e9ecef;}
        .stTabs [aria-selected="true"] {background-color: #6e8efb;color: white;}
        .stDataFrame {border-radius: 8px;box-shadow: 0 2px 4px rgba(0,0,0,0.1);border: 1px solid #e0e0e0;}
        .stExpander {border-radius: 8px;box-shadow: 0 2px 4px rgba(0,0,0,0.1);border: 1px solid #e0e0e0;}
        .stExpander .streamlit-expanderHeader {font-weight: 600;color: #2c3e50;}
        ::-webkit-scrollbar {width: 8px;height: 8px;}
        ::-webkit-scrollbar-track {background: #f1f1f1;border-radius: 10px;}
        ::-webkit-scrollbar-thumb {background: #6e8efb;border-radius: 10px;}
        ::-webkit-scrollbar-thumb:hover {background: #5a7de3;}
        
        /* Entity type color coding */
        .entity-hospital {background-color: #1f77b4; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;}
        .entity-doctor {background-color: #ff7f0e; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;}
        .entity-clinic {background-color: #1f77b4; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;}
        .entity-pharmacy {background-color: #2ca02c; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;}
        .entity-location {background-color: #e377c2; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;}
        .entity-service_time {background-color: #bcbd22; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;}
        
        /* Card styles */
        .info-card {background-color: white;padding: 1.5rem;border-radius: 8px;box-shadow: 0 2px 4px rgba(0,0,0,0.1);margin-bottom: 1rem;}
        .metric-card {background: linear-gradient(135deg, #6e8efb, #a777e3);color: white;padding: 1rem;border-radius: 8px;text-align: center;}
        
        /* Attribute nodes in graph */
        .attribute-node {background-color: #8c564b; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7rem;}
    </style>
    """, unsafe_allow_html=True)
    
    init_database()
    
    # Main header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1f77b4, #6e8efb); padding: 2rem; border-radius: 0 0 8px 8px; margin-bottom: 2rem;box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="font-size: 2.5rem;">🏥</div>
            <div>
                <h1 style="color: white; margin: 0;">AI-Powered Healthcare Knowledge Graph Explorer</h1>
                <p style="color: rgba(255,255,255,0.9); margin: 0;">Extract and visualize relationships between hospitals, clinics, and doctors using AI</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0; color: white;">🔧 Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        model_option = st.selectbox(
            "Select Gemini Model",
            ["gemini-2.5-flash"],
            index=0,
            help="Select the Gemini model for entity extraction"
        )
        
        graph_type = st.selectbox(
            "Graph Visualization Type",
            ["interactive", "plotly", "simple"],
            index=0,
            help="Choose how to visualize the knowledge graph"
        )
        
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0; color: white;">📊 System Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            stored_df = get_stored_entities()
            if not stored_df.empty:
                entities = stored_df[stored_df['entity_type'].isin(['hospital', 'clinic', 'doctor'])]
                total_entities = len(stored_df)
                healthcare_count = len(entities)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Entities", total_entities)
                with col2:
                    st.metric("Healthcare Entities", healthcare_count)
                
                # Entity type breakdown
                entity_counts = stored_df['entity_type'].value_counts()
                st.markdown("**Entity Distribution:**")
                for entity_type, count in entity_counts.head(5).items():
                    type_color = get_entity_type_color(entity_type)
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span class="entity-{entity_type.lower().replace('_', '-')}">{entity_type}</span>
                        <span><strong>{count}</strong></span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.metric("Data Source", "SQLite Database")
            else:
                st.info("No entities stored yet")
        except:
            st.info("Database not initialized")
        
        if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
            if st.checkbox("I understand this will delete all data"):
                try:
                    conn = sqlite3.connect('entities.db')
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM entities')
                    cursor.execute('DELETE FROM relationship_edges')
                    cursor.execute('DELETE FROM entity_attributes')
                    conn.commit()
                    conn.close()
                    st.success("All data cleared successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing data: {e}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📥 Extract Entities", "🌐 View Knowledge Graph", "🔍 Analyze Relationships"])
    
    with tab1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #6e8efb, #a777e3); padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin: 0; display: flex; align-items: center; gap: 10px;">📥 Extract Healthcare Entities</h2>
        </div>
        """, unsafe_allow_html=True)
        
        input_method = st.radio(
            "Select Input Method",
            ["Text", "URL", "File"],
            horizontal=True,
            key="input_method"
        )
        
        source_type = "Text"
        source_identifier = "Manual Input"
        
        if input_method == "Text":
            text_input = st.text_area(
                "Enter healthcare-related text:",
                height=200,
                key="text_input",
                placeholder="""Paste text about hospitals, clinics, doctors, medical services, etc. here...

Example:
'City General Hospital located at 123 Main Street offers 24/7 emergency services with pharmacy and ambulance. 
Dr. Smith specializes in cardiology with 15 years experience, 85% success rate, and consultation fees of $200.
Open Monday to Friday from 9:00 AM to 6:00 PM.'"""
            )
            
            if st.button("🚀 Extract Entities", key="extract_text", type="primary", use_container_width=True):
                if text_input:
                    with st.spinner("Extracting healthcare entities using AI..."):
                        process_input_with_model(text_input, source_type, source_identifier, model_option)
                else:
                    st.error("Please enter some text to extract entities from.")
        
        elif input_method == "URL":
            url_input = st.text_input(
                "Enter healthcare website URL:",
                key="url_input",
                placeholder="https://example-hospital.com/services"
            )
            
            if st.button("🚀 Extract from URL", key="extract_url", type="primary", use_container_width=True):
                if url_input:
                    source_type = "URL"
                    source_identifier = url_input
                    with st.spinner("Extracting text from URL and processing..."):
                        text = extract_text_from_url(url_input)
                        if text:
                            process_input_with_model(text, source_type, source_identifier, model_option)
                        else:
                            st.error("Failed to extract text from the URL.")
                else:
                    st.error("Please enter a valid URL.")
                    
        elif input_method == "File":
            uploaded_file = st.file_uploader(
                "Upload healthcare document",
                type=['txt', 'pdf', 'docx', 'csv'],
                help="Supported formats: .txt, .pdf, .docx, .csv"
            )
            
            if uploaded_file and st.button("🚀 Extract from File", type="primary", use_container_width=True):
                try:
                    if uploaded_file.type == "text/plain":
                        text = str(uploaded_file.read(), "utf-8")
                    elif uploaded_file.type == "text/csv":
                        df = pd.read_csv(uploaded_file)
                        text = df.to_string()
                    else:
                        st.warning("Please upload a text file (.txt) or CSV (.csv) for now")
                        text = ""
                    
                    if text:
                        with st.spinner("Processing document..."):
                            process_input_with_model(text, "file", uploaded_file.name, model_option)
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        # Display stored entities with enhanced view
        stored_df = get_stored_entities()
        if not stored_df.empty:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #6e8efb, #a777e3); padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h2 style="color: white; margin: 0; display: flex; align-items: center; gap: 10px;">📚 Stored Entities</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                doctors = len(stored_df[stored_df['entity_type'] == 'doctor'])
                st.metric("👨‍⚕️ Doctors", doctors)
            with col2:
                hospitals = len(stored_df[stored_df['entity_type'].isin(['hospital', 'clinic'])])
                st.metric("🏥 Hospitals/Clinics", hospitals)
            with col3:
                locations = len(stored_df[stored_df['entity_type'].isin(['location', 'address'])])
                st.metric("📍 Locations", locations)
            with col4:
                other = len(stored_df[~stored_df['entity_type'].isin(['doctor', 'hospital', 'clinic', 'location', 'address'])])
                st.metric("📊 Other Entities", other)
            
            # Filters and search
            col1, col2 = st.columns(2)
            with col1:
                entity_type_filter = st.multiselect(
                    "Filter by Entity Type",
                    ["All"] + sorted(stored_df['entity_type'].unique().tolist()),
                    default=["All"],
                    key="entity_type_filter_stored"
                )
            with col2:
                search_query = st.text_input(
                    "Search Entities",
                    key="entity_search",
                    placeholder="Search by name, attributes, or relationships"
                )
            
            filtered_df = stored_df.copy()
            if "All" not in entity_type_filter:
                filtered_df = filtered_df[filtered_df['entity_type'].isin(entity_type_filter)]
            if search_query:
                filtered_df = search_entities(search_query, filtered_df)
            
            if not filtered_df.empty:
                # Enhanced entity display with detailed view option
                display_option = st.radio(
                    "Display Mode",
                    ["Detailed Cards", "Compact Table"],
                    horizontal=True,
                    key="display_option"
                )
                
                if display_option == "Detailed Cards":
                    # Group by entity type
                    entity_types = filtered_df['entity_type'].unique()
                    
                    for entity_type in entity_types:
                        type_df = filtered_df[filtered_df['entity_type'] == entity_type]
                        type_color = get_entity_type_color(entity_type)
                        
                        with st.expander(f"{entity_type.capitalize()} ({len(type_df)})", expanded=True):
                            # Display in a grid
                            cols = st.columns(2)
                            for idx, (_, row) in enumerate(type_df.iterrows()):
                                col_idx = idx % 2
                                with cols[col_idx]:
                                    # Create a card for each entity
                                    st.markdown(f"""
                                    <div class="info-card">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                            <h4 style="margin: 0; color: {type_color};">{row['entity_name']}</h4>
                                            <span class="entity-{entity_type.lower().replace('_', '-')}">{entity_type}</span>
                                        </div>
                                        <div style="margin-bottom: 0.5rem;">
                                            <small><strong>Weight:</strong> {row['weight']:.2f}</small>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Show key attributes
                                    detailed_attrs = safe_json_loads(row.get('detailed_attributes', '{}'))
                                    if detailed_attrs:
                                        for key, value in list(detailed_attrs.items())[:3]:  # Show first 3 attributes
                                            if value and value != 'Not specified':
                                                st.markdown(f"<small><strong>{key.replace('_', ' ').title()}:</strong> {value}</small>", unsafe_allow_html=True)
                                    
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        if st.button("View", key=f"view_card_{row['entity_id']}", use_container_width=True):
                                            st.session_state.selected_node = row['entity_id']
                                            st.rerun()
                                    with col_b:
                                        if st.button("Add to Graph", key=f"add_card_{row['entity_id']}", use_container_width=True):
                                            if row['entity_id'] not in st.session_state.graph_nodes:
                                                st.session_state.graph_nodes.add(row['entity_id'])
                                                st.session_state.entity_types[row['entity_id']] = entity_type
                                                st.session_state.node_colors[row['entity_id']] = type_color
                                                st.session_state.node_visibility[row['entity_id']] = True
                                                st.success(f"Added {row['entity_name']} to graph")
                                            else:
                                                st.info(f"{row['entity_name']} already in graph")
                                    
                                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    # Compact table view
                    def color_entity_type(val):
                        color_map = {
                            'hospital': '#1f77b4',
                            'clinic': '#1f77b4',
                            'doctor': '#ff7f0e',
                            'pharmacy': '#2ca02c',
                            'ambulance_service': '#d62728',
                            'person': '#9467bd',
                            'organization': '#8c564b',
                            'location': '#e377c2',
                            'general': '#7f7f7f',
                            'nurse': '#17becf',
                            'service_time': '#bcbd22',
                            'address': '#7f7f7f'
                        }
                        color = color_map.get(val.lower(), '#7f7f7f')
                        return f'background-color: {color}; color: white;'
                    
                    display_df = filtered_df[['entity_id', 'entity_name', 'entity_type', 'attributes', 'relationships', 'weight']]
                    
                    try:
                        styled_df = display_df.style.map(color_entity_type, subset=['entity_type'])
                        st.dataframe(
                            styled_df,
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "entity_id": st.column_config.NumberColumn("ID"),
                                "entity_name": st.column_config.TextColumn("Name"),
                                "entity_type": st.column_config.TextColumn("Type"),
                                "attributes": st.column_config.TextColumn("Attributes"),
                                "relationships": st.column_config.TextColumn("Relationships"),
                                "weight": st.column_config.NumberColumn("Weight", format="%.2f")
                            }
                        )
                    except KeyError:
                        st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    with tab2:
        visualize_knowledge_graph()
    
    with tab3:
        analyze_entity_relationships()

if __name__ == "__main__":
    main()