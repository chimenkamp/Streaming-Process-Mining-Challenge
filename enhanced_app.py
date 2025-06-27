import inspect
import os
import zipfile
import shutil
import tempfile
import uuid
from pathlib import Path
import base64
import io

import dash
from dash import dcc, html, Input, Output, State, callback, clientside_callback, ClientsideFunction
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import importlib.util
from typing import Dict, List, Any, Optional, Type
import traceback

from src.functions import tumbling_window_average

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'ui'))

# Import our custom components and classes
from src.ui.algorithm_base import BaseAlgorithm, EventStream
from src.ui.styles import get_nord_styles, NORD_COLORS, get_custom_css
from src.ui.stream_manager import StreamManager

# Initialize Dash app with assets folder
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                assets_folder='assets')
app.title = "Streaming Process Mining Challenge"

# Add custom CSS to the app
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
''' + get_custom_css() + '''
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Get Nord styling
nord_styles = get_nord_styles()

# Initialize stream manager with correct config path
stream_manager = StreamManager()

# Create uploads directory
UPLOAD_DIRECTORY = os.path.join(os.getcwd(), "uploaded_algorithms")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

def save_uploaded_file(contents, filename, session_id):
    """Save uploaded file and return the file path."""
    if contents is None:
        return None
    
    # Decode the file content
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Create session directory
    session_dir = os.path.join(UPLOAD_DIRECTORY, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    file_path = os.path.join(session_dir, filename)
    
    # Save the file
    if filename.endswith('.zip'):
        # Handle ZIP file
        with open(file_path, 'wb') as f:
            f.write(decoded)
        
        # Extract ZIP contents
        extract_dir = os.path.join(session_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        return extract_dir
    else:
        # Handle Python file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(decoded.decode('utf-8'))
        
        return file_path

def find_algorithm_files(directory_path):
    """Find Python files containing algorithm implementations."""
    algorithm_files = []
    
    if os.path.isfile(directory_path) and directory_path.endswith('.py'):
        return [directory_path]
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                # Check if file likely contains an algorithm
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'BaseAlgorithm' in content or 'class' in content:
                            algorithm_files.append(file_path)
                except:
                    continue
    
    return algorithm_files

def load_algorithm_from_file(file_path):
    """Load algorithm class from Python file."""
    try:
        # Add the file's directory to Python path
        file_dir = os.path.dirname(file_path)
        if file_dir not in sys.path:
            sys.path.insert(0, file_dir)
        
        # Import the module
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        
        # Add necessary imports to module namespace
        module.BaseAlgorithm = BaseAlgorithm
        module.Dict = Dict
        module.Any = Any
        module.List = List
        module.pd = pd
        module.time = __import__('time')
        
        spec.loader.exec_module(module)
        
        # Find algorithm class
        algorithm_class = find_concrete_algorithm_class(module.__dict__, BaseAlgorithm)
        
        return algorithm_class, None
    except Exception as e:
        return None, str(e)

# Layout
app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Streaming Process Mining Challenge",
                    className="text-center mb-4",
                    style={'color': NORD_COLORS['snow_storm'][2], 'font-weight': 'bold'})
        ])
    ], className="mb-4"),

    # Main content - 50/50 layout
    dbc.Row([
        # Left side - Algorithm Upload (50%)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Algorithm Submission",
                            style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0})
                ], style=nord_styles['card_header']),
                dbc.CardBody([
                    # Upload status display
                    html.Div(id='upload-status', children=[
                        html.Div([
                            html.I(className="fas fa-cloud-upload-alt", 
                                   style={'font-size': '48px', 'color': NORD_COLORS['snow_storm'][0], 'margin-bottom': '20px'}),
                            html.H5("No Algorithm Uploaded", 
                                    style={'color': NORD_COLORS['snow_storm'][1], 'margin-bottom': '10px'}),
                            html.P("Upload a Python file (.py) or ZIP archive (.zip) containing your algorithm implementation.",
                                   style={'color': NORD_COLORS['snow_storm'][0], 'text-align': 'center', 'margin-bottom': '20px'}),
                            dbc.Button(
                                "📁 Upload Algorithm",
                                id="upload-btn",
                                color="primary",
                                size="lg",
                                style={
                                    'background-color': NORD_COLORS['frost'][2],
                                    'border-color': NORD_COLORS['frost'][2],
                                    'border-radius': '8px',
                                    'width': '100%',
                                    'margin-bottom': '15px'
                                }
                            )
                        ], style={'text-align': 'center', 'padding': '40px 20px'})
                    ]),
                    
                    # Algorithm info (hidden initially)
                    html.Div(id='algorithm-info', style={'display': 'none'}, children=[
                        html.Div(id='algorithm-details'),
                        html.Hr(style={'border-color': NORD_COLORS['polar_night'][3]}),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "🔄 Upload New Algorithm",
                                    id="reupload-btn",
                                    color="outline-primary",
                                    size="sm",
                                    style={
                                        'border-color': NORD_COLORS['frost'][2],
                                        'color': NORD_COLORS['frost'][2],
                                        'border-radius': '6px'
                                    }
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Button(
                                    "🧪 Test Algorithm",
                                    id="test-uploaded-btn",
                                    color="success",
                                    size="sm",
                                    disabled=True,
                                    style={
                                        'background-color': NORD_COLORS['aurora'][3],
                                        'border-color': NORD_COLORS['aurora'][3],
                                        'border-radius': '6px',
                                        'width': '100%'
                                    }
                                )
                            ], width=6)
                        ])
                    ])
                ], style=nord_styles['card_body'])
            ], style=nord_styles['card'])
        ], width=6),

        # Right side - Controls and Results (50%)
        dbc.Col([
            # Test Stream Selection
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Test Configuration",
                            style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0})
                ], style=nord_styles['card_header']),
                dbc.CardBody([
                    html.Label("Select Test Stream:",
                               style={'color': NORD_COLORS['snow_storm'][1], 'font-weight': 'bold'}),
                    dcc.Dropdown(
                        id='stream-selector',
                        options=stream_manager.get_testing_streams(),
                        value=None,
                        placeholder="Select a stream...",
                        className='dash-dropdown',
                        style={
                            'background-color': NORD_COLORS['polar_night'][1],
                            'color': NORD_COLORS['snow_storm'][2],
                            'border-radius': '6px'
                        }
                    ),
                    html.Br(),
                    html.Div(id='stream-description',
                             style={'color': NORD_COLORS['snow_storm'][0],
                                    'font-size': '12px',
                                    'font-style': 'italic',
                                    'margin-bottom': '15px',
                                    'padding': '10px',
                                    'background-color': NORD_COLORS['polar_night'][2],
                                    'border-radius': '4px',
                                    'white-space': 'pre-line',
                                    'display': 'none'}),
                    html.Label("Number of Cases:",
                               style={'color': NORD_COLORS['snow_storm'][1], 'font-weight': 'bold'}),
                    dcc.Slider(
                        id='num-cases-slider',
                        min=50,
                        max=1000,
                        step=50,
                        value=500,
                        marks={i: str(i) for i in range(50, 1001, 200)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        disabled=True
                    ),
                    html.Br(),
                    dbc.Button(
                        "Run Test",
                        id="run-test-btn",
                        color="success",
                        disabled=True,
                        style={
                            'background-color': NORD_COLORS['aurora'][3],
                            'border-color': NORD_COLORS['aurora'][3],
                            'border-radius': '6px',
                            'width': '100%'
                        }
                    )
                ], style=nord_styles['card_body'])
            ], style={**nord_styles['card'], 'margin-bottom': '20px'}),

            # Stream Scrubber
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Stream Timeline",
                            style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0})
                ], style=nord_styles['card_header']),
                dbc.CardBody([
                    html.Div([
                        html.Div(id='stream-scrubber', className='stream-scrubber', children=[
                            html.Div(className='stream-scrubber-track', children=[
                                html.Div(id='warmup-marker', className='warmup-marker'),
                                html.Div(id='drift-marker', className='drift-marker', style={'display': 'none'}),
                                html.Div(id='scrubber-progress', className='stream-scrubber-progress'),
                            ]),
                            html.Div(id='scrubber-handle', className='stream-scrubber-handle')
                        ]),
                        html.Div(id='timeline-info', className='timeline-info', children=[
                            html.Span("Select a stream to see timeline"),
                            html.Span(id='phase-indicator', className='phase-indicator phase-learning',
                                      children=""),
                            html.Span("")
                        ])
                    ], style={'opacity': '0.3'})
                ], style=nord_styles['card_body'], id='timeline-container')
            ], style={**nord_styles['card'], 'margin-bottom': '20px'}),

            # Results Display
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.H5("Conformance Results",
                                style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0, 'flex': '1'}),
                        dbc.Button(
                            "📊 Fullscreen",
                            id="fullscreen-btn",
                            size="sm",
                            color="primary",
                            style={
                                'background-color': NORD_COLORS['frost'][2],
                                'border-color': NORD_COLORS['frost'][2],
                                'border-radius': '6px',
                                'font-size': '12px',
                                'display': 'none'
                            }
                        )
                    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between'})
                ], style=nord_styles['card_header']),
                dbc.CardBody([
                    html.Div([
                        html.P("Upload an algorithm and run a test to see results",
                               style={'color': NORD_COLORS['snow_storm'][0], 'text-align': 'center',
                                      'margin': '50px 0'})
                    ], id='results-placeholder'),
                    dcc.Graph(
                        id='results-plot',
                        style={'height': '350px', 'display': 'none'}
                    ),
                    html.Div(id='metrics-display', style={'margin-top': '10px'})
                ], style=nord_styles['card_body'], id='results-container')
            ], style={**nord_styles['card'], 'margin-bottom': '20px'}),

            # Submission Form
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Submit to Challenge",
                            style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0})
                ], style=nord_styles['card_header']),
                dbc.CardBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Label("Team Name:", style={'color': NORD_COLORS['snow_storm'][1]}),
                            dbc.Input(
                                id="team-name",
                                type="text",
                                placeholder="Enter your team name",
                                className="form-control"
                            )
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Label("Email:", style={'color': NORD_COLORS['snow_storm'][1]}),
                            dbc.Input(
                                id="email",
                                type="email",
                                placeholder="contact@example.com",
                                className="form-control"
                            )
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Label("Description:", style={'color': NORD_COLORS['snow_storm'][1]}),
                            dbc.Textarea(
                                id="description",
                                placeholder="Brief description of your approach...",
                                className="form-control",
                                style={'height': '80px'}
                            )
                        ], className="mb-3"),
                        dbc.Button(
                            "Submit Algorithm",
                            id="submit-btn",
                            color="primary",
                            disabled=True,
                            style={
                                'background-color': NORD_COLORS['frost'][2],
                                'border-color': NORD_COLORS['frost'][2],
                                'border-radius': '6px',
                                'width': '100%'
                            }
                        )
                    ])
                ], style=nord_styles['card_body'])
            ], style=nord_styles['card'])
        ], width=6)
    ]),

    # File Upload Modal
    dbc.Modal([
        dbc.ModalHeader([
            dbc.ModalTitle("Upload Algorithm",
                           style={'color': NORD_COLORS['snow_storm'][2]})
        ], style={'background-color': NORD_COLORS['polar_night'][1],
                  'border-bottom': f'1px solid {NORD_COLORS["polar_night"][3]}'}),
        dbc.ModalBody([
            html.Div([
                html.H6("File Upload", style={'color': NORD_COLORS['snow_storm'][2], 'margin-bottom': '15px'}),
                dcc.Upload(
                    id='algorithm-upload',
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt", 
                               style={'font-size': '24px', 'margin-right': '10px', 'color': NORD_COLORS['frost'][2]}),
                        'Drag and Drop or ',
                        html.A('Select Files', style={'color': NORD_COLORS['frost'][2], 'text-decoration': 'underline'})
                    ]),
                    style={
                        'width': '100%',
                        'height': '80px',
                        'lineHeight': '80px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '8px',
                        'borderColor': NORD_COLORS['polar_night'][3],
                        'textAlign': 'center',
                        'backgroundColor': NORD_COLORS['polar_night'][2],
                        'color': NORD_COLORS['snow_storm'][1],
                        'cursor': 'pointer',
                        'margin-bottom': '20px'
                    },
                    accept='.py,.zip',
                    multiple=False
                ),
                
                html.P("Supported formats: Python files (.py) or ZIP archives (.zip)",
                       style={'color': NORD_COLORS['snow_storm'][0], 'font-size': '12px', 'margin-bottom': '20px'}),
                
                html.Hr(style={'border-color': NORD_COLORS['polar_night'][3]}),
                
                html.H6("Required Libraries (Optional)", style={'color': NORD_COLORS['snow_storm'][2], 'margin-bottom': '15px'}),
                dbc.Textarea(
                    id="required-libraries",
                    placeholder="List required Python libraries, one per line:\nnumpy\npandas\nscikit-learn",
                    style={'height': '100px', 'margin-bottom': '15px'},
                    className="form-control"
                ),
                
                html.Div(id='upload-validation', style={'margin-top': '15px'})
            ])
        ], style={'background-color': NORD_COLORS['polar_night'][1], 'padding': '30px'}),
        dbc.ModalFooter([
            dbc.Button(
                "Cancel",
                id="upload-cancel-btn",
                color="secondary",
                style={
                    'background-color': NORD_COLORS['polar_night'][3],
                    'border-color': NORD_COLORS['polar_night'][3],
                    'color': NORD_COLORS['snow_storm'][2],
                    'margin-right': '10px'
                }
            ),
            dbc.Button(
                "Upload Algorithm",
                id="upload-confirm-btn",
                color="primary",
                disabled=True,
                style={
                    'background-color': NORD_COLORS['frost'][2],
                    'border-color': NORD_COLORS['frost'][2],
                    'color': NORD_COLORS['snow_storm'][2]
                }
            )
        ], style={'background-color': NORD_COLORS['polar_night'][1],
                  'border-top': f'1px solid {NORD_COLORS["polar_night"][3]}'})
    ], id="upload-modal",
        size="lg",
        is_open=False),

    # Fullscreen Plot Modal
    dbc.Modal([
        dbc.ModalHeader([
            dbc.ModalTitle("Conformance Analysis - Fullscreen View",
                           style={'color': NORD_COLORS['snow_storm'][2]})
        ], style={'background-color': NORD_COLORS['polar_night'][1],
                  'border-bottom': f'1px solid {NORD_COLORS["polar_night"][3]}'}),
        dbc.ModalBody([
            dcc.Graph(
                id='fullscreen-plot',
                style={'height': '75vh'}
            ),
            html.Div(id='fullscreen-metrics', style={'margin-top': '20px'})
        ], style={'background-color': NORD_COLORS['polar_night'][1], 'padding': '30px'}),
        dbc.ModalFooter([
            dbc.Button(
                "Close",
                id="close-fullscreen-btn",
                color="secondary",
                style={
                    'background-color': NORD_COLORS['polar_night'][3],
                    'border-color': NORD_COLORS['polar_night'][3],
                    'color': NORD_COLORS['snow_storm'][2]
                }
            )
        ], style={'background-color': NORD_COLORS['polar_night'][1],
                  'border-top': f'1px solid {NORD_COLORS["polar_night"][3]}'})
    ], id="fullscreen-modal",
        size="xl",
        style={'max-width': '90vw', 'max-height': '90vh'},
        is_open=False),

    # Hidden div to store session data
    html.Div(id='session-id', style={'display': 'none'}, children=str(uuid.uuid4())),
    html.Div(id='test-results-store', style={'display': 'none'}),
    html.Div(id='stream-data-store', style={'display': 'none'}),
    html.Div(id='current-position-store', style={'display': 'none'}, children='0'),
    html.Div(id='uploaded-algorithm-store', style={'display': 'none'})

], fluid=True, style={
    'background-color': NORD_COLORS['polar_night'][0],
    'min-height': '100vh',
    'padding': '20px'
})

# Callback to open upload modal
@app.callback(
    Output("upload-modal", "is_open"),
    [Input("upload-btn", "n_clicks"), 
     Input("reupload-btn", "n_clicks"),
     Input("upload-cancel-btn", "n_clicks"), 
     Input("upload-confirm-btn", "n_clicks")],
    [State("upload-modal", "is_open")],
)
def toggle_upload_modal(upload_clicks, reupload_clicks, cancel_clicks, confirm_clicks, is_open):
    if upload_clicks or reupload_clicks or cancel_clicks or confirm_clicks:
        return not is_open
    return is_open

# Callback to handle file upload validation
@app.callback(
    [Output('upload-validation', 'children'),
     Output('upload-confirm-btn', 'disabled')],
    [Input('algorithm-upload', 'contents')],
    [State('algorithm-upload', 'filename')]
)
def validate_upload(contents, filename):
    if contents is None:
        return "", True
    
    if filename:
        if filename.endswith(('.py', '.zip')):
            validation_msg = dbc.Alert(
                f"✓ File '{filename}' is ready for upload",
                color="success",
                style={'background-color': f'{NORD_COLORS["aurora"][3]}20',
                       'border-color': NORD_COLORS['aurora'][3],
                       'color': NORD_COLORS['aurora'][3]}
            )
            return validation_msg, False
        else:
            validation_msg = dbc.Alert(
                f"✗ Invalid file type. Please upload a .py or .zip file",
                color="danger",
                style={'background-color': f'{NORD_COLORS["aurora"][0]}20',
                       'border-color': NORD_COLORS['aurora'][0],
                       'color': NORD_COLORS['aurora'][0]}
            )
            return validation_msg, True
    
    return "", True

# Callback to process uploaded algorithm
@app.callback(
    [Output('upload-status', 'style'),
     Output('algorithm-info', 'style'),
     Output('algorithm-details', 'children'),
     Output('uploaded-algorithm-store', 'children'),
     Output('test-uploaded-btn', 'disabled'),
     Output('submit-btn', 'disabled')],
    [Input('upload-confirm-btn', 'n_clicks')],
    [State('algorithm-upload', 'contents'),
     State('algorithm-upload', 'filename'),
     State('required-libraries', 'value'),
     State('session-id', 'children')]
)
def process_uploaded_algorithm(n_clicks, contents, filename, libraries, session_id):
    if not n_clicks or contents is None:
        return (
            {'display': 'block'}, {'display': 'none'}, "", "", True, True
        )
    
    try:
        # Save uploaded file
        file_path = save_uploaded_file(contents, filename, session_id)
        
        if file_path is None:
            raise Exception("Failed to save uploaded file")
        
        # Find algorithm files
        algorithm_files = find_algorithm_files(file_path)
        
        if not algorithm_files:
            raise Exception("No Python files with algorithm implementations found")
        
        # Try to load algorithm from the first valid file
        algorithm_class = None
        error_messages = []
        
        for algo_file in algorithm_files:
            try:
                algorithm_class, error = load_algorithm_from_file(algo_file)
                if algorithm_class:
                    break
                if error:
                    error_messages.append(f"{os.path.basename(algo_file)}: {error}")
            except Exception as e:
                error_messages.append(f"{os.path.basename(algo_file)}: {str(e)}")
        
        if not algorithm_class:
            error_detail = "\n".join(error_messages) if error_messages else "No valid algorithm class found"
            raise Exception(f"Could not load algorithm class. {error_detail}")
        
        # Create algorithm info display
        algorithm_info = html.Div([
            html.Div([
                html.I(className="fas fa-check-circle", 
                       style={'color': NORD_COLORS['aurora'][3], 'margin-right': '10px', 'font-size': '20px'}),
                html.H5(f"Algorithm Loaded: {algorithm_class.__name__}", 
                        style={'color': NORD_COLORS['aurora'][3], 'display': 'inline-block', 'margin': 0})
            ], style={'margin-bottom': '15px'}),
            
            html.Div([
                html.Strong("File:", style={'color': NORD_COLORS['snow_storm'][1]}),
                html.Span(f" {filename}", style={'color': NORD_COLORS['snow_storm'][2], 'margin-left': '5px'})
            ], style={'margin-bottom': '8px'}),
            
            html.Div([
                html.Strong("Files Found:", style={'color': NORD_COLORS['snow_storm'][1]}),
                html.Span(f" {len(algorithm_files)} Python file(s)", 
                         style={'color': NORD_COLORS['snow_storm'][2], 'margin-left': '5px'})
            ], style={'margin-bottom': '8px'}),
            
            html.Div([
                html.Strong("Class Name:", style={'color': NORD_COLORS['snow_storm'][1]}),
                html.Span(f" {algorithm_class.__name__}", 
                         style={'color': NORD_COLORS['snow_storm'][2], 'margin-left': '5px'})
            ], style={'margin-bottom': '8px'}),
            
            html.Div([
                html.Strong("Required Libraries:", style={'color': NORD_COLORS['snow_storm'][1]}),
                html.Div(
                    libraries.strip() if libraries else "None specified",
                    style={'color': NORD_COLORS['snow_storm'][2], 'margin-top': '5px', 
                           'background-color': NORD_COLORS['polar_night'][2], 'padding': '8px',
                           'border-radius': '4px', 'font-family': 'monospace', 'font-size': '12px'}
                )
            ], style={'margin-bottom': '15px'})
        ])
        
        # Store algorithm information
        algorithm_data = {
            'class_name': algorithm_class.__name__,
            'file_path': file_path,
            'filename': filename,
            'libraries': libraries,
            'algorithm_files': algorithm_files,
            'session_id': session_id
        }
        
        return (
            {'display': 'none'},  # Hide upload status
            {'display': 'block'},  # Show algorithm info
            algorithm_info,
            json.dumps(algorithm_data),
            False,  # Enable test button
            False   # Enable submit button
        )
        
    except Exception as e:
        error_info = html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle", 
                       style={'color': NORD_COLORS['aurora'][0], 'margin-right': '10px', 'font-size': '20px'}),
                html.H5("Upload Failed", 
                        style={'color': NORD_COLORS['aurora'][0], 'display': 'inline-block', 'margin': 0})
            ], style={'margin-bottom': '15px'}),
            
            dbc.Alert(
                f"Error: {str(e)}",
                color="danger",
                style={'background-color': f'{NORD_COLORS["aurora"][0]}20',
                       'border-color': NORD_COLORS['aurora'][0],
                       'color': NORD_COLORS['aurora'][0]}
            )
        ])
        
        return (
            {'display': 'none'},  # Hide upload status
            {'display': 'block'},  # Show error info
            error_info,
            "",
            True,  # Disable test button
            True   # Disable submit button
        )

# Update run test button to work with uploaded algorithms
@app.callback(
    Output('run-test-btn', 'disabled'),
    [Input('stream-selector', 'value'),
     Input('uploaded-algorithm-store', 'children')]
)
def update_run_test_button(stream_id, algorithm_data):
    has_stream = stream_id is not None
    has_algorithm = algorithm_data and algorithm_data != ""
    return not (has_stream and has_algorithm)

# Callback to enable/disable controls based on stream selection
@app.callback(
    [Output('stream-description', 'children'),
     Output('stream-description', 'style'),
     Output('num-cases-slider', 'disabled'),
     Output('timeline-container', 'style')],
    [Input('stream-selector', 'value')]
)
def update_controls_on_stream_selection(stream_id):
    base_description_style = {
        'color': NORD_COLORS['snow_storm'][0],
        'font-size': '12px',
        'font-style': 'italic',
        'margin-bottom': '15px',
        'padding': '10px',
        'background-color': NORD_COLORS['polar_night'][2],
        'border-radius': '4px',
        'white-space': 'pre-line'
    }

    base_timeline_style = nord_styles['card_body']

    if not stream_id:
        return (
            "No stream selected",
            {**base_description_style, 'display': 'none'},
            True,  # disable slider
            {**base_timeline_style, 'opacity': '0.3'}  # dim timeline
        )

    description = stream_manager.get_stream_description(stream_id)
    return (
        description,
        {**base_description_style, 'display': 'block'},
        False,  # enable slider
        base_timeline_style  # normal timeline
    )

def find_concrete_algorithm_class(
        exec_globals: Dict[str, Any], base_class: Type[BaseAlgorithm]
) -> Optional[Type[BaseAlgorithm]]:
    """
    Scan exec_globals to find a concrete subclass of base_class.
    """
    for obj in exec_globals.values():
        if not inspect.isclass(obj):
            continue

        if obj is base_class:
            continue

        if not issubclass(obj, base_class):
            continue

        abstracts = getattr(obj, "__abstractmethods__", set())
        if abstracts:
            continue

        return obj

    return None

def calculate_drift_positions_from_stream(event_log, drift_config=None):
    """Calculate actual drift start and end positions based on concept:origin column and drift configuration."""
    if 'concept:origin' not in event_log.columns:
        return {'has_drift': False, 'drift_start': 0, 'drift_end': 0}

    if drift_config:
        drift_type = drift_config.get('drift_type', 'sudden')
        drift_begin_percentage = drift_config.get('drift_begin_percentage', 0.0)
        drift_end_percentage = drift_config.get('drift_end_percentage', 0.0)

        total_events = len(event_log)

        if drift_type == 'gradual' and drift_begin_percentage != drift_end_percentage:
            drift_start = int(drift_begin_percentage * total_events)
            drift_end = int(drift_end_percentage * total_events)

            return {
                'has_drift': True,
                'drift_start': drift_start,
                'drift_end': drift_end,
                'drift_type': drift_type
            }
        elif drift_type == 'sudden':
            drift_point = int(drift_begin_percentage * total_events)

            return {
                'has_drift': True,
                'drift_start': drift_point,
                'drift_end': drift_point,
                'drift_type': drift_type
            }

    # Fallback to origin-based detection
    origins = event_log['concept:origin'].tolist()

    drift_start = None
    for i, origin in enumerate(origins):
        if origin == "Log B":
            drift_start = i
            break

    if drift_start is None:
        return {'has_drift': False, 'drift_start': 0, 'drift_end': 0}

    total_events = len(origins)
    log_b_count_in_first_half = sum(
        1 for i in range(drift_start, min(drift_start + (total_events - drift_start) // 2, total_events))
        if origins[i] == "Log B")
    log_a_count_in_second_half = sum(1 for i in range(drift_start + (total_events - drift_start) // 2, total_events)
                                     if origins[i] == "Log A")

    if log_b_count_in_first_half > 0 and log_a_count_in_second_half > 0:
        window_size = max(20, total_events // 20)

        for i in range(drift_start + window_size, total_events - window_size, window_size // 2):
            window_origins = origins[i:i + window_size]
            log_b_ratio = sum(1 for origin in window_origins if origin == "Log B") / len(window_origins)

            if log_b_ratio >= 0.8:
                drift_end = i
                break
        else:
            last_log_a = None
            for i in range(drift_start, len(origins)):
                if origins[i] == "Log A":
                    last_log_a = i

            drift_end = last_log_a + 1 if last_log_a is not None else len(origins)
    else:
        drift_end = len(origins)

        last_log_a = None
        for i in range(drift_start, len(origins)):
            if origins[i] == "Log A":
                last_log_a = i

        if last_log_a is not None:
            drift_end = last_log_a + 1

    return {
        'has_drift': True,
        'drift_start': drift_start,
        'drift_end': drift_end,
        'drift_type': 'gradual' if log_b_count_in_first_half > 0 and log_a_count_in_second_half > 0 else 'sudden'
    }

# Updated run algorithm test callback to work with uploaded files
@app.callback(
    [Output('results-plot', 'figure'),
     Output('results-plot', 'style'),
     Output('results-placeholder', 'style'),
     Output('metrics-display', 'children'),
     Output('test-results-store', 'children'),
     Output('stream-data-store', 'children'),
     Output('timeline-info', 'children'),
     Output('warmup-marker', 'style'),
     Output('drift-marker', 'style'),
     Output('fullscreen-btn', 'style')],
    [Input('run-test-btn', 'n_clicks')],
    [State('stream-selector', 'value'),
     State('num-cases-slider', 'value'),
     State('uploaded-algorithm-store', 'children')]
)
def run_algorithm_test(n_clicks, stream_type, num_cases, algorithm_data_json):
    if not n_clicks or not stream_type or not algorithm_data_json:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=NORD_COLORS['polar_night'][1],
            plot_bgcolor=NORD_COLORS['polar_night'][1],
            font=dict(color=NORD_COLORS['snow_storm'][2])
        )
        return (
            fig,
            {'height': '350px', 'display': 'none'},
            {'color': NORD_COLORS['snow_storm'][0], 'text-align': 'center', 'margin': '50px 0', 'display': 'block'},
            html.Div(),
            "{}",
            "{}",
            [html.Span("Upload an algorithm and select a stream"), html.Span(""), html.Span("")],
            {'display': 'none'},
            {'display': 'none'},
            {'background-color': NORD_COLORS['frost'][2], 'border-color': NORD_COLORS['frost'][2],
             'border-radius': '6px', 'font-size': '12px', 'display': 'none'}
        )

    try:
        algorithm_data = json.loads(algorithm_data_json)
        
        # Load algorithm from uploaded files
        algorithm_files = algorithm_data['algorithm_files']
        algorithm_class = None
        
        for algo_file in algorithm_files:
            algorithm_class, error = load_algorithm_from_file(algo_file)
            if algorithm_class:
                break
        
        if not algorithm_class:
            raise ValueError("Could not load algorithm class from uploaded files")

        # Generate concept drift stream using StreamManager
        event_log, baseline_results = stream_manager.generate_concept_drift_stream(stream_type, num_cases)
        event_stream = EventStream(event_log)

        # Get stream info for drift configuration
        stream_info = stream_manager.get_stream_info(stream_type)
        drift_config = stream_info.get('drift_config') if stream_info else None

        # Calculate actual drift positions from the event stream WITH drift config
        drift_data = calculate_drift_positions_from_stream(event_log, drift_config)

        # Initialize algorithm
        algorithm = algorithm_class()

        # Learning phase
        ground_truth_events = event_stream.get_ground_truth_events()
        algorithm.on_learning_phase_start({'total_events': len(ground_truth_events)})

        for event in ground_truth_events:
            algorithm.learn(event)

        algorithm.on_learning_phase_end({'learned_events': len(ground_truth_events)})

        # Conformance checking phase
        full_stream = event_stream.get_full_stream()
        conformance_results = []

        for i, event in enumerate(full_stream):
            conformance_score = algorithm.conformance(event)
            conformance_results.append(conformance_score)

        # Calculate global errors against baseline
        global_errors = sum(1 for i, (alg, base) in enumerate(zip(conformance_results, baseline_results))
                            if abs(alg - base) > 0.3)

        # Create comparison plot with actual drift data
        fig = create_conformance_plot(conformance_results, baseline_results, len(ground_truth_events),
                                      stream_type, drift_data)

        # Get final metrics
        performance_metrics = algorithm.get_performance_metrics()
        metrics_display = create_metrics_display(performance_metrics, global_errors, baseline_results,
                                                 conformance_results)

        # Store results
        test_results = {
            'algorithm_results': conformance_results,
            'baseline_results': baseline_results,
            'performance_metrics': performance_metrics,
            'global_errors': global_errors,
            'learning_split': len(ground_truth_events),
            'drift_data': drift_data,
            'algorithm_info': algorithm_data
        }

        stream_data = {
            'total_events': len(full_stream),
            'learning_events': len(ground_truth_events),
            'learning_ratio': len(ground_truth_events) / len(full_stream),
            'stream_id': stream_type,
            'drift_data': drift_data
        }

        # Timeline info
        timeline_info = [
            html.Span(f"Event: 0 / {len(full_stream)}"),
            html.Span("Learning Phase", className='phase-indicator phase-learning'),
            html.Span("Progress: 0%")
        ]

        # Warmup marker position
        warmup_position = (len(ground_truth_events) / len(full_stream)) * 100
        warmup_style = {
            'left': f'{warmup_position}%',
            'display': 'block'
        }

        # Drift marker using actual drift positions
        drift_style = {'display': 'none'}
        if drift_data.get('has_drift', False):
            drift_position = (drift_data['drift_start'] / len(full_stream)) * 100
            drift_style = {
                'left': f'{drift_position}%',
                'display': 'block',
                'background': NORD_COLORS['aurora'][0],
                'position': 'absolute',
                'top': '0',
                'width': '2px',
                'height': '100%',
                'z-index': '3'
            }

            if drift_data.get('drift_type') == 'gradual' and drift_data.get('drift_end', 0) > drift_data.get('drift_start', 0):
                drift_start_pos = (drift_data['drift_start'] / len(full_stream)) * 100
                drift_end_pos = (drift_data['drift_end'] / len(full_stream)) * 100
                drift_width = drift_end_pos - drift_start_pos

                drift_style = {
                    'left': f'{drift_start_pos}%',
                    'width': f'{drift_width}%',
                    'display': 'block',
                    'background': f'linear-gradient(to right, {NORD_COLORS["aurora"][0]}80, {NORD_COLORS["aurora"][0]})',
                    'position': 'absolute',
                    'top': '0',
                    'height': '100%',
                    'z-index': '3',
                    'border-left': f'2px solid {NORD_COLORS["aurora"][0]}',
                    'border-right': f'2px solid {NORD_COLORS["aurora"][0]}'
                }

        # Show fullscreen button
        fullscreen_btn_style = {
            'background-color': NORD_COLORS['frost'][2],
            'border-color': NORD_COLORS['frost'][2],
            'border-radius': '6px',
            'font-size': '12px',
            'display': 'inline-block'
        }

        return (
            fig,
            {'height': '350px', 'display': 'block'},
            {'display': 'none'},
            metrics_display,
            json.dumps(test_results, default=str),
            json.dumps(stream_data),
            timeline_info,
            warmup_style,
            drift_style,
            fullscreen_btn_style
        )

    except Exception as e:
        # Error handling
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(color=NORD_COLORS['aurora'][0], size=16)
        )
        error_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=NORD_COLORS['polar_night'][1],
            plot_bgcolor=NORD_COLORS['polar_night'][1]
        )

        error_display = dbc.Alert(
            f"Error: {str(e)}",
            color="danger",
            style={'background-color': NORD_COLORS['aurora'][0]}
        )

        return (
            error_fig,
            {'height': '350px', 'display': 'block'},
            {'display': 'none'},
            error_display,
            "{}",
            "{}",
            [html.Span("Event: 0 / 0"), html.Span("Error", className='phase-indicator'), html.Span("Progress: 0%")],
            {'display': 'none'},
            {'display': 'none'},
            {'background-color': NORD_COLORS['frost'][2], 'border-color': NORD_COLORS['frost'][2],
             'border-radius': '6px', 'font-size': '12px', 'display': 'none'}
        )

# Fullscreen modal callbacks
@app.callback(
    Output("fullscreen-modal", "is_open"),
    [Input("fullscreen-btn", "n_clicks"), Input("close-fullscreen-btn", "n_clicks")],
    [State("fullscreen-modal", "is_open")],
)
def toggle_fullscreen_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

@app.callback(
    [Output('fullscreen-plot', 'figure'),
     Output('fullscreen-metrics', 'children')],
    [Input('fullscreen-btn', 'n_clicks')],
    [State('test-results-store', 'children'),
     State('stream-data-store', 'children')]
)
def update_fullscreen_plot(n_clicks, test_results_json, stream_data_json):
    if not n_clicks or not test_results_json or test_results_json == "{}":
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=NORD_COLORS['polar_night'][1],
            plot_bgcolor=NORD_COLORS['polar_night'][1],
            font=dict(color=NORD_COLORS['snow_storm'][2]),
            title="No data available"
        )
        return fig, html.Div()

    try:
        test_results = json.loads(test_results_json)
        stream_data = json.loads(stream_data_json)

        user_results = test_results['algorithm_results']
        baseline_results = test_results['baseline_results']
        learning_split = test_results['learning_split']
        stream_id = stream_data.get('stream_id')

        drift_data = test_results.get('drift_data', None)

        fig = create_conformance_plot(user_results, baseline_results, learning_split, stream_id, drift_data)

        fig.update_layout(
            height=600,
            title={
                'text': "Conformance Analysis - Detailed View",
                'x': 0.5,
                'font': {'size': 20, 'color': NORD_COLORS['snow_storm'][2]}
            },
            font={'size': 14},
            margin=dict(l=60, r=60, t=80, b=60)
        )

        performance_metrics = test_results['performance_metrics']
        global_errors = test_results['global_errors']
        metrics_display = create_metrics_display(performance_metrics, global_errors, baseline_results, user_results)

        return fig, metrics_display

    except Exception as e:
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error loading fullscreen view: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(color=NORD_COLORS['aurora'][0], size=16)
        )
        error_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=NORD_COLORS['polar_night'][1],
            plot_bgcolor=NORD_COLORS['polar_night'][1]
        )
        return error_fig, html.Div()

# Stream scrubber interaction
clientside_callback(
    """
    function(clickData, streamData, currentPos) {
        if (!streamData || !clickData || clickData.length === 0) {
            return [0, ["Event: 0 / 0", "Learning Phase", "Progress: 0%"], "phase-learning", "0"];
        }

        const data = JSON.parse(streamData);
        const totalEvents = data.total_events || 0;
        const learningEvents = data.learning_events || 0;

        if (totalEvents === 0) return [0, ["Event: 0 / 0", "Learning Phase", "Progress: 0%"], "phase-learning", "0"];

        const rect = document.getElementById('stream-scrubber').getBoundingClientRect();
        const clickX = clickData.clientX - rect.left;
        const scrubberWidth = rect.width - 20;
        const position = Math.max(0, Math.min(1, (clickX - 10) / scrubberWidth));

        const currentEvent = Math.floor(position * totalEvents);
        const progress = (currentEvent / totalEvents * 100).toFixed(1);

        const isLearningPhase = currentEvent < learningEvents;
        const phase = isLearningPhase ? "Learning Phase" : "Conformance Phase";
        const phaseClass = isLearningPhase ? "phase-learning" : "phase-conformance";

        return [
            position * 100,
            [
                `Event: ${currentEvent} / ${totalEvents}`,
                phase,
                `Progress: ${progress}%`
            ],
            phaseClass,
            (position * 100).toString()
        ];
    }
    """,
    [Output('scrubber-progress', 'style'),
     Output('timeline-info', 'children', allow_duplicate=True),
     Output('phase-indicator', 'className', allow_duplicate=True),
     Output('current-position-store', 'children', allow_duplicate=True)],
    [Input('stream-scrubber', 'n_clicks')],
    [State('stream-data-store', 'children'),
     State('current-position-store', 'children')],
    prevent_initial_call=True
)

@app.callback(
    Output('submit-btn', 'children'),
    [Input('submit-btn', 'n_clicks')],
    [State('team-name', 'value'),
     State('email', 'value'),
     State('description', 'value'),
     State('uploaded-algorithm-store', 'children'),
     State('test-results-store', 'children')]
)
def submit_algorithm(n_clicks, team_name, email, description, algorithm_data, test_results):
    if not n_clicks:
        return "Submit Algorithm"

    if not all([team_name, email, description]):
        return "Please fill all fields"

    if not algorithm_data or algorithm_data == "":
        return "Please upload an algorithm first"

    # Here you would implement actual submission logic
    # For now, just simulate successful submission
    return "✓ Submitted Successfully!"

def create_conformance_plot(user_results, baseline_results, learning_split, stream_id=None, drift_data=None):
    """Create a conformance comparison plot with learning phase indicator and actual drift positions."""
    fig = go.Figure()

    timestamps = list(range(len(user_results)))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=baseline_results,
        mode='lines',
        name='Baseline (Ground Truth)',
        line=dict(color=NORD_COLORS['aurora'][1], width=2, dash='dash'),
        opacity=0.8,
        hovertemplate='<b>Baseline</b><br>Event: %{x}<br>Conformance: %{y:.3f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=user_results,
        mode='lines',
        name='Your Algorithm',
        line=dict(color=NORD_COLORS['frost'][2], width=3),
        hovertemplate='<b>Algorithm</b><br>Event: %{x}<br>Conformance: %{y:.3f}<extra></extra>'
    ))

    fig.add_vline(
        x=learning_split,
        line_width=2,
        line_dash="solid",
        line_color=NORD_COLORS['aurora'][2],
        annotation_text="Learning Phase End",
        annotation_position="top"
    )

    fig.add_vrect(
        x0=0, x1=learning_split,
        fillcolor=NORD_COLORS['aurora'][2],
        opacity=0.1,
        layer="below",
        line_width=0,
    )

    if drift_data and drift_data.get('has_drift', False):
        drift_start = drift_data['drift_start']
        drift_end = drift_data['drift_end']

        fig.add_vrect(
            x0=drift_start, x1=drift_end,
            fillcolor=NORD_COLORS['aurora'][0],
            opacity=0.15,
            layer="below",
            line_width=0,
        )

        fig.add_vline(
            x=drift_start,
            line_width=2,
            line_dash="dot",
            line_color=NORD_COLORS['aurora'][0],
            annotation_text="Concept Drift Start",
            annotation_position="top"
        )

        if drift_end < len(user_results):
            fig.add_vline(
                x=drift_end,
                line_width=2,
                line_dash="dot",
                line_color=NORD_COLORS['aurora'][0],
                annotation_text="Concept Drift End",
                annotation_position="bottom"
            )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=NORD_COLORS['polar_night'][1],
        plot_bgcolor=NORD_COLORS['polar_night'][1],
        font=dict(color=NORD_COLORS['snow_storm'][2]),
        title="Conformance Score vs Event Index",
        xaxis_title="Event Index",
        yaxis_title="Conformance Score",
        yaxis=dict(range=[0, 1]),
        legend=dict(
            bgcolor=NORD_COLORS['polar_night'][2],
            bordercolor=NORD_COLORS['polar_night'][3],
            borderwidth=1,
            font=dict(color=NORD_COLORS['snow_storm'][2])
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

def create_metrics_display(metrics, global_errors, baseline_results, user_results):
    """Create a display for performance metrics."""
    metric_cards = []

    if len(baseline_results) > 0 and len(user_results) > 0:
        mae = sum(abs(a - b) for a, b in zip(user_results, baseline_results)) / len(user_results)
        rmse = (sum((a - b) ** 2 for a, b in zip(user_results, baseline_results)) / len(user_results)) ** 0.5
        accurate_predictions = sum(1 for a, b in zip(user_results, baseline_results) if abs(a - b) <= 0.1)
        accuracy = accurate_predictions / len(user_results)

        additional_metrics = {
            'Mean Absolute Error': mae,
            'Root Mean Square Error': rmse,
            'Accuracy (±0.1)': accuracy,
            'Global Errors': global_errors
        }
    else:
        additional_metrics = {'Global Errors': global_errors}

    all_metrics = {**additional_metrics, **metrics}

    for metric, value in all_metrics.items():
        if isinstance(value, float):
            if 'error' in metric.lower() or 'mae' in metric.lower() or 'rmse' in metric.lower():
                display_value = f"{value:.4f}"
                color = NORD_COLORS['aurora'][0] if value > 0.2 else NORD_COLORS['aurora'][3]
            elif 'accuracy' in metric.lower():
                display_value = f"{value:.1%}"
                color = NORD_COLORS['aurora'][3] if value > 0.8 else NORD_COLORS['aurora'][2]
            else:
                display_value = f"{value:.4f}"
                color = NORD_COLORS['frost'][2]
        else:
            display_value = str(value)
            color = NORD_COLORS['aurora'][0] if value > 0 else NORD_COLORS['aurora'][3]

        card = dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6(metric.replace('_', ' ').title(),
                            style={'color': NORD_COLORS['snow_storm'][1], 'margin': 0, 'font-size': '12px'}),
                    html.H5(display_value,
                            style={'color': color, 'margin': 0})
                ], style={'padding': '8px', 'text-align': 'center'})
            ], style={
                'background-color': NORD_COLORS['polar_night'][2],
                'border': 'none',
                'border-radius': '6px'
            })
        ], width=12 // min(len(all_metrics), 4))
        metric_cards.append(card)

    return dbc.Row(metric_cards)

if __name__ == '__main__':
    app.run_server(debug=True)