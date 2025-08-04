import inspect
import os
import traceback
import zipfile
import uuid
import base64
import sqlite3
import json
from datetime import datetime, timedelta

import dash
from dash import dcc, html, Input, Output, State, callback_context, clientside_callback, ClientsideFunction, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import sys
import importlib.util
from typing import Dict, List, Any, Optional, Type

from dash._utils import logger

from src.ui.leaderboard_system import SubmissionManager, LeaderboardManager
from src.ui.leaderboard_ui import create_leaderboard_layout, create_leaderboard_table, create_performance_chart, \
    create_recent_activity_list

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'ui'))

# Import our custom components and classes
from src.ui.algorithm_base import BaseAlgorithm
from src.ui.algorithm_executor import AlgorithmExecutor, ExecutionResult

from src.ui.styles import get_nord_styles, NORD_COLORS, get_custom_css
from src.ui.stream_manager import StreamManager

# Import enhanced components (create simplified versions if files don't exist)

from src.ui.database_manager import DatabaseManager, db_manager
from src.ui.algorithm_testing_engine import AlgorithmTestingEngine, validate_algorithm_file

ENHANCED_FEATURES = True

LEADERBOARD_AVAILABLE = True

# Initialize Dash app with assets folder
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                assets_folder='assets',
                suppress_callback_exceptions=True)
app.title = "AVOCADO: The Streaming Process Mining Challenge"

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
    <body style="background-color: rgb(46, 52, 64); color: #D8DEE9; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">
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

# Initialize managers
stream_manager = StreamManager()

if LEADERBOARD_AVAILABLE:
    submission_manager = SubmissionManager("uploaded_algorithms")
    submission_manager.set_stream_manager(stream_manager)
else:
    submission_manager = SubmissionManager("uploaded_algorithms")

# Create uploads directory
UPLOAD_DIRECTORY = os.path.join(os.getcwd(), "uploaded_algorithms")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Use enhanced database manager with default algorithm support
user_db = db_manager
algorithm_executor = AlgorithmExecutor(BaseAlgorithm, stream_manager, db_manager)


# Utility functions
def generate_user_token():
    return str(uuid.uuid4())


def calculate_file_hash(content):
    import hashlib
    return hashlib.sha256(content.encode()).hexdigest()


def save_uploaded_file(contents, filename, token):
    if contents is None:
        return None

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    user_dir = os.path.join(UPLOAD_DIRECTORY, token)
    os.makedirs(user_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(filename)
    unique_filename = f"{name}_{timestamp}{ext}"
    file_path = os.path.join(user_dir, unique_filename)

    if filename.endswith('.zip'):
        with open(file_path, 'wb') as f:
            f.write(decoded)
        extract_dir = os.path.join(user_dir, f'extracted_{timestamp}')
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return extract_dir
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(decoded.decode('utf-8'))
        return file_path


# Create main navigation
def create_navigation():
    return dbc.Tabs([
        dbc.Tab(label="📤 Upload Algorithm", tab_id="upload", active_tab_style={'fontWeight': 'bold'}),
        dbc.Tab(label="🧪 Test Algorithm", tab_id="test", active_tab_style={'fontWeight': 'bold'}),
        dbc.Tab(label="📊 My Algorithms", tab_id="my-algorithms", active_tab_style={'fontWeight': 'bold'}),
        dbc.Tab(label="🏆 Leaderboard", tab_id="leaderboard", active_tab_style={'fontWeight': 'bold'}),
        dbc.Tab(label="📚 Documentation", tab_id="documentation", active_tab_style={'fontWeight': 'bold'})
    ], id="main-tabs", active_tab="my-algorithms",
        style={'marginBottom': '20px', 'color': NORD_COLORS['snow_storm'][2]})


# Create upload layout
def create_upload_layout():
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Upload Your Own Algorithm",
                            style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0})
                ], style=nord_styles['card_header']),
                dbc.CardBody([
                    dbc.Alert([
                        html.H6("💡 Getting Started", style={'margin': 0, 'color': NORD_COLORS['frost'][2]}),
                        html.P(
                            "New to the platform? Check out the default algorithms in the 'My Algorithms' tab to see examples and test them first!",
                            style={'margin': '10px 0 0 0', 'color': NORD_COLORS['snow_storm'][1]})
                    ], color="info", style={'backgroundColor': f'{NORD_COLORS["frost"][2]}20',
                                            'borderColor': NORD_COLORS['frost'][2],
                                            'marginBottom': '20px'}),

                    dcc.Upload(
                        id='algorithm-file-upload',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt",
                                   style={'fontSize': '48px', 'color': NORD_COLORS['frost'][2],
                                          'marginBottom': '20px'}),
                            html.H5("Drag & Drop or Click to Upload",
                                    style={'color': NORD_COLORS['snow_storm'][1], 'marginBottom': '10px'}),
                            html.P(
                                "Upload a Python file (.py) or ZIP archive (.zip) containing your algorithm implementation.",
                                style={'color': NORD_COLORS['snow_storm'][0], 'textAlign': 'center',
                                       'marginBottom': '20px'}),
                            html.P("Supported formats: .py, .zip | Max size: 50MB",
                                   style={'color': NORD_COLORS['snow_storm'][0], 'fontSize': '12px'})
                        ], style={'textAlign': 'center', 'padding': '60px 20px'}),
                        style={
                            'width': '100%',
                            'minHeight': '300px',
                            'lineHeight': '60px',
                            'borderWidth': '3px',
                            'borderStyle': 'dashed',
                            'borderRadius': '12px',
                            'borderColor': NORD_COLORS['polar_night'][3],
                            'textAlign': 'center',
                            'backgroundColor': NORD_COLORS['polar_night'][2],
                            'cursor': 'pointer',
                            'marginBottom': '20px',
                            'transition': 'all 0.3s ease'
                        },
                        accept='.py,.zip',
                        multiple=False
                    ),
                    html.Div(id='upload-file-status', style={'marginBottom': '20px'}),
                    html.Hr(style={'borderColor': NORD_COLORS['polar_night'][3]}),
                    html.H6("Required Libraries (Optional)",
                            style={'color': NORD_COLORS['snow_storm'][2], 'marginBottom': '15px'}),
                    dbc.Textarea(
                        id="algorithm-libraries",
                        placeholder="List required Python libraries, one per line:\nnumpy>=1.21.0\npandas>=1.3.0\nscikit-learn",
                        style={'height': '120px', 'marginBottom': '15px'},
                        className="form-control"
                    ),
                    html.P("Libraries will be validated during testing.",
                           style={'color': NORD_COLORS['snow_storm'][0], 'fontSize': '12px'})
                ], style=nord_styles['card_body'])
            ], style=nord_styles['card'])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Algorithm Information",
                            style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0})
                ], style=nord_styles['card_header']),
                dbc.CardBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Label("Algorithm Name *",
                                      style={'color': NORD_COLORS['snow_storm'][1], 'fontWeight': 'bold'}),
                            dbc.Input(
                                id="upload-algorithm-name",
                                type="text",
                                placeholder="e.g., Advanced Conformance Checker v2.1",
                                className="form-control",
                                style={'marginBottom': '15px'}
                            )
                        ], className="mb-3"),

                        dbc.Row([
                            dbc.Label("Email Address *",
                                      style={'color': NORD_COLORS['snow_storm'][1], 'fontWeight': 'bold'}),
                            dbc.Input(
                                id="upload-email",
                                type="email",
                                placeholder="your.email@university.edu",
                                className="form-control",
                                style={'marginBottom': '15px'}
                            )
                        ], className="mb-3"),

                        dbc.Row([
                            dbc.Label("Institution",
                                      style={'color': NORD_COLORS['snow_storm'][1], 'fontWeight': 'bold'}),
                            dbc.Input(
                                id="upload-institution",
                                type="text",
                                placeholder="University Name, Company, etc.",
                                className="form-control",
                                style={'marginBottom': '15px'}
                            )
                        ], className="mb-3"),

                        dbc.Row([
                            dbc.Label("Algorithm Description",
                                      style={'color': NORD_COLORS['snow_storm'][1], 'fontWeight': 'bold'}),
                            dbc.Textarea(
                                id="upload-description",
                                placeholder="Brief description of your algorithm approach, techniques used, key features, etc.",
                                style={'height': '100px', 'marginBottom': '15px'},
                                className="form-control"
                            )
                        ], className="mb-3"),

                        html.Hr(style={'borderColor': NORD_COLORS['polar_night'][3]}),

                        dbc.Button(
                            "🚀 Upload Algorithm",
                            id="confirm-upload-btn",
                            color="primary",
                            size="lg",
                            disabled=True,
                            style={
                                'backgroundColor': NORD_COLORS['frost'][2],
                                'borderColor': NORD_COLORS['frost'][2],
                                'borderRadius': '8px',
                                'width': '100%',
                                'marginBottom': '15px'
                            }
                        ),

                        html.Div(id='upload-success-message', style={'marginTop': '15px'})
                    ])
                ], style=nord_styles['card_body'])
            ], style=nord_styles['card'])
        ], width=6)
    ])


# Create test layout
def create_test_layout():
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Select Algorithm to Test",
                            style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0})
                ], style=nord_styles['card_header']),
                dbc.CardBody([
                    # This area will be populated by the callback
                    html.Div(id='algorithm-selection-area'),
                ], style=nord_styles['card_body'])
            ], style=nord_styles['card'])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Test Configuration",
                            style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0})
                ], style=nord_styles['card_header']),
                dbc.CardBody([
                    html.Label("Select Test Stream:",
                               style={'color': NORD_COLORS['snow_storm'][1], 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='test-stream-selector',
                        options=stream_manager.get_testing_streams(),
                        value=None,
                        placeholder="Select a stream...",
                        className='dash-dropdown'
                    ),
                    html.Br(),
                    # html.Div(id='test-stream-description', style={'display': 'none'}),
                    html.Label("Number of Cases:",
                               style={'color': NORD_COLORS['snow_storm'][1], 'fontWeight': 'bold'}),
                    dcc.Slider(
                        id='test-num-cases-slider',
                        min=50,
                        max=1000,
                        step=50,
                        value=500,
                        marks={i: str(i) for i in range(50, 1001, 200)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Br(),
                    dbc.Button(
                        "🧪 Run Test",
                        id="run-algorithm-test-btn",
                        color="success",
                        disabled=True,
                        style={
                            'backgroundColor': NORD_COLORS['aurora'][3],
                            'borderColor': NORD_COLORS['aurora'][3],
                            'borderRadius': '6px',
                            'width': '100%'
                        }
                    )
                ], style=nord_styles['card_body'])
            ], style={**nord_styles['card'], 'marginBottom': '20px'}),

            dbc.Card([
                dbc.CardHeader([
                    html.H5("Test Results",
                            style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0})
                ], style=nord_styles['card_header']),
                dbc.CardBody([
                    html.Div([
                        html.P("Select an algorithm and run a test to see results",
                               style={'color': NORD_COLORS['snow_storm'][0], 'textAlign': 'center',
                                      'margin': '50px 0'})
                    ], id='test-results-placeholder'),
                    dcc.Graph(
                        id='test-results-plot',
                        style={'height': '350px', 'display': 'none'}
                    ),
                    html.Div(id='test-metrics-display', style={'margin-top': '10px'})
                ], style=nord_styles['card_body'])
            ], style=nord_styles['card'])
        ], width=6)
    ])


# Create my algorithms layout with enhanced default algorithm support
def create_my_algorithms_layout():
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H5("📊 My Algorithms",
                        style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0})
            ], style=nord_styles['card_header']),
            dbc.CardBody([
                dbc.Alert([
                    html.H6("🎯 Welcome to AVOCADO!", style={'margin': 0, 'color': NORD_COLORS['aurora'][3]}),
                    html.P(
                        "Get started by testing the default algorithms below. They demonstrate how to implement conformance checking algorithms for the platform.",
                        style={'margin': '10px 0 0 0', 'color': NORD_COLORS['snow_storm'][1]})
                ], color="success", style={'backgroundColor': f'{NORD_COLORS["aurora"][3]}20',
                                           'borderColor': NORD_COLORS['aurora'][3],
                                           'marginBottom': '20px'}),

                html.Div(id="user-algorithms-list"),
                html.Hr(style={'borderColor': NORD_COLORS['polar_night'][3]}),
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            "🏆 Submit Best Algorithm to Challenge",
                            id="submit-to-challenge-btn",
                            color="primary",
                            disabled=True,
                            style={
                                'backgroundColor': NORD_COLORS['frost'][2],
                                'borderColor': NORD_COLORS['frost'][2],
                                'borderRadius': '6px'
                            }
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Button(
                            "📤 Upload Your Own Algorithm",
                            id="goto-upload-from-myalgs-btn",
                            color="outline-primary",
                            style={
                                'borderColor': NORD_COLORS['frost'][2],
                                'color': NORD_COLORS['frost'][2],
                                'borderRadius': '6px',
                                'width': '100%'
                            }
                        )
                    ], width=6)
                ])
            ], style=nord_styles['card_body'])
        ], style=nord_styles['card'])
    ])


# Create documentation layout
def create_documentation_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("📚 Documentation", style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0})
                    ], style=nord_styles['card_header']),
                    dbc.CardBody([
                        dcc.Markdown("""
# Streaming Process Mining Challenge

## Getting Started

**New users start with default example algorithms!** When you first visit the platform, you'll find pre-loaded example algorithms in your "My Algorithms" tab that demonstrate how to implement conformance checking algorithms.

### Quick Start Guide

1. **Explore Examples**: Go to "My Algorithms" to see the default algorithms
2. **Test Examples**: Use the "Test Algorithm" tab to run the examples on different streams  
3. **Upload Your Own**: Create your own algorithm and upload it via the "Upload Algorithm" tab
4. **Submit to Challenge**: Submit your best performing algorithm to the leaderboard

## Algorithm Requirements

Your algorithm must implement these methods:

```python
class MyAlgorithm(BaseAlgorithm):
    def learn(self, event: Dict[str, Any]) -> None:
        # Learn from event during warm-up phase
        pass

    def conformance(self, event: Dict[str, Any]) -> float:
        # Return conformance score (0.0 to 1.0)
        pass
```

## Event Format

Events are dictionaries with these keys:
- `case:concept:name`: Case identifier
- `concept:name`: Activity name  
- `time:timestamp`: Event timestamp
- `concept:origin`: Origin log (Only in testing)

## Default Algorithms

The platform provides two example algorithms:

### 1. Frequency-Based Conformance Checker
- **Approach**: Learns activity transition frequencies during learning phase
- **Conformance**: Uses transition probabilities to assess conformance
- **Good for**: Understanding the basic approach to conformance checking

### 2. Simple Baseline Algorithm
- **Approach**: Basic rule-based conformance checking
- **Conformance**: Returns high scores for activities seen during learning
- **Good for**: Understanding the minimal implementation requirements

## Evaluation Metrics

- **Accuracy**: Percentage of predictions within 0.1 of baseline
- **MAE**: Mean Absolute Error compared to baseline
- **RMSE**: Root Mean Square Error
- **Speed**: Average processing time per event
- **Robustness**: Number of major errors (>0.3 difference)

## Test Streams

The platform provides several concept drift streams:
- **Simple Concept Drift**: Sudden change at 60% of stream
- **Gradual Concept Drift**: Gradual transition between 30-70%
- **Early Sudden Drift**: Concept drift at 20% of stream

## Workflow

1. **Start with Defaults**: Test the provided example algorithms
2. **Understand Performance**: Analyze how different approaches perform
3. **Develop Your Own**: Create improved algorithms based on insights
4. **Test and Iterate**: Use the testing environment to refine your approach
5. **Submit to Challenge**: Submit your best algorithm to compete

## Tips for Success

- Start by understanding how the default algorithms work
- Test on different streams to understand concept drift
- Focus on both accuracy and speed
- Use the learning phase effectively to build robust models
- Consider adaptive approaches for concept drift scenarios

## Support

If you encounter issues or have questions:
- Check the default algorithm implementations for reference
- Test your algorithms thoroughly before submission
- Review the evaluation metrics to understand scoring
                        """, style={'color': NORD_COLORS['snow_storm'][2]})
                    ], style=nord_styles['card_body'])
                ], style=nord_styles['card'])
            ])
        ])
    ])


# Main Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("AVOCADO",
                    className="text-center mb-0",
                    style={'color': NORD_COLORS['aurora'][2], 'fontWeight': 'bold'})
        ])
    ], className="mb-1"),
    dbc.Row([
        dbc.Col([
            html.H3("The Streaming Process Mining Challenge",
                    className="text-center",
                    style={'fontWeight': 'bold', 'color': NORD_COLORS['aurora'][4]})
        ])
    ], className="mb-4"),

    # Navigation
    create_navigation(),

    # Tab Content
    html.Div(id="tab-content"),

    # Hidden components for data storage
    html.Div(id='user-token-store', style={'display': 'none'}),
    html.Div(id='selected-algorithm-store', style={'display': 'none'}),
    html.Div(id='test-results-store', style={'display': 'none'}),
    html.Div(id='algorithm-refresh-trigger', style={'display': 'none'}, children='0'),

    # Interval component for token management
    dcc.Interval(
        id='interval-component',
        interval=5000,
        n_intervals=0
    ),

], fluid=True, style={
    'backgroundColor': NORD_COLORS['polar_night'][0],
    'minHeight': '100vh',
    'padding': '20px'
})


# FIXED CALLBACKS - Using pattern matching properly

# Tab switching callback
@app.callback(
    Output("tab-content", "children"),
    [Input("main-tabs", "active_tab")]
)
def render_tab_content(active_tab):
    if active_tab == "upload":
        return create_upload_layout()
    elif active_tab == "test":
        return create_test_layout()
    elif active_tab == "my-algorithms":
        return create_my_algorithms_layout()
    elif active_tab == "leaderboard":
        return create_leaderboard_layout(NORD_COLORS)
    elif active_tab == "documentation":
        return create_documentation_layout()
    else:
        return html.Div("Select a tab")


# Enhanced client-side callback for user token management and default algorithm seeding
app.clientside_callback(
    """
    function(n_intervals) {
        try {
            let token = localStorage.getItem('avocado_user_token');
            if (!token) {
                token = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                    const r = Math.random() * 16 | 0;
                    const v = c === 'x' ? r : (r & 0x3 | 0x8);
                    return v.toString(16);
                });
                localStorage.setItem('avocado_user_token', token);
            }
            return token;
        } catch (e) {
            return 'session_' + Math.random().toString(36).substr(2, 9);
        }
    }
    """,
    Output('user-token-store', 'children'),
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=False
)


# Enhanced callback to automatically create user and seed defaults
@app.callback(
    Output('algorithm-refresh-trigger', 'children', allow_duplicate=True),
    [Input('user-token-store', 'children')],
    prevent_initial_call=True
)
def ensure_user_exists_with_defaults(token):
    """Ensure user exists and has default algorithms seeded."""
    if not token:
        return dash.no_update

    try:
        # This will create user if not exists and seed default algorithms
        user_db.create_or_update_user(token, email=f"user_{token[:8]}@platform.local")

        # Return timestamp to trigger refresh
        import time
        return str(int(time.time()))
    except Exception as e:
        logger.error(f"Error ensuring user exists: {e}")
        return dash.no_update


# File upload validation callback
@app.callback(
    [Output('upload-file-status', 'children'),
     Output('confirm-upload-btn', 'disabled')],
    [Input('algorithm-file-upload', 'contents'),
     Input('upload-algorithm-name', 'value'),
     Input('upload-email', 'value')],
    [State('algorithm-file-upload', 'filename')]
)
def validate_upload_form(contents, algorithm_name, email, filename):
    if not contents:
        return "", True

    if not algorithm_name or not email:
        return dbc.Alert("Please fill in algorithm name and email", color="warning"), True

    if not filename or not (filename.endswith('.py') or filename.endswith('.zip')):
        return dbc.Alert("Only .py and .zip files are supported", color="danger"), True

    if '@' not in email:
        return dbc.Alert("Please enter a valid email address", color="warning"), True

    return dbc.Alert(f"✓ Ready to upload {filename}", color="success"), False


# Upload confirmation callback
@app.callback(
    [Output('upload-success-message', 'children'),
     Output('upload-file-status', 'children', allow_duplicate=True),
     Output('algorithm-file-upload', 'contents'),
     Output('upload-algorithm-name', 'value'),
     Output('upload-institution', 'value'),
     Output('upload-description', 'value'),
     Output('algorithm-libraries', 'value')],
    [Input('confirm-upload-btn', 'n_clicks')],
    [State('algorithm-file-upload', 'contents'),
     State('algorithm-file-upload', 'filename'),
     State('upload-algorithm-name', 'value'),
     State('upload-email', 'value'),
     State('upload-institution', 'value'),
     State('upload-description', 'value'),
     State('algorithm-libraries', 'value'),
     State('user-token-store', 'children')],
    prevent_initial_call=True
)
def handle_algorithm_upload(n_clicks, contents, filename, algorithm_name,
                            email, institution, description, libraries, token):
    if not n_clicks or not contents:
        return "", "", None, "", "", "", ""

    try:
        user_db.create_or_update_user(token, email, institution)

        file_path = save_uploaded_file(contents, filename, token)
        if not file_path:
            return dbc.Alert("Failed to save file", color="danger"), "", None, "", "", "", ""

        content_type, content_string = contents.split(',')
        file_hash = calculate_file_hash(content_string)
        file_size = len(base64.b64decode(content_string))

        required_libs = []
        if libraries:
            required_libs = [lib.strip() for lib in libraries.split('\n') if lib.strip()]

        algorithm_id = user_db.add_algorithm(
            token=token,
            algorithm_name=algorithm_name,
            filename=filename,
            file_path=file_path,
            file_hash=file_hash,
            description=description,
            required_libraries=required_libs,
            file_size=file_size,
            is_default=False  # User-uploaded algorithms are not defaults
        )

        if algorithm_id:
            success_msg = dbc.Alert([
                html.H5("✅ Upload Successful!", style={'margin': 0}),
                html.P(f"Algorithm '{algorithm_name}' has been uploaded and validated.",
                       style={'margin-bottom': '10px'}),
                html.P(f"Algorithm ID: {algorithm_id} | File size: {file_size // 1024:.1f} KB",
                       style={'margin': 0, 'font-size': '12px', 'opacity': '0.8'})
            ], color="success")

            return success_msg, "", None, "", "", "", ""
        else:
            return dbc.Alert("Failed to save algorithm to database", color="danger"), "", None, "", "", "", ""

    except Exception as e:
        return dbc.Alert(f"Upload failed: {str(e)}", color="danger"), "", None, "", "", "", ""


@app.callback(
    Output('algorithm-selection-area', 'children'),
    [Input('main-tabs', 'active_tab'),
     Input('algorithm-refresh-trigger', 'children'),
     Input('selected-algorithm-store', 'children')],
    [State('user-token-store', 'children')]
)
def load_user_algorithms_for_testing(active_tab: str, refresh_trigger: str, selected_algorithm_id: str,
                                     token: str) -> html.Div:
    """
    Loads the user's algorithms, highlighting the selected one and showing enhanced info for defaults.
    """
    if active_tab != "test":
        return html.Div()

    if not token:
        return html.Div(html.P("Loading algorithms...", style={'color': NORD_COLORS['snow_storm'][0]}))

    try:
        selected_id: Optional[int] = int(selected_algorithm_id) if selected_algorithm_id else None
        algorithms: list = user_db.get_user_algorithms(token)
    except (ValueError, TypeError):
        selected_id = None
    except Exception as e:
        return html.Div(html.P(f"Error loading algorithms: {str(e)}", style={'color': NORD_COLORS['aurora'][0]}))

    if not algorithms:
        return html.Div([
            html.P("No algorithms available yet.",
                   style={'color': NORD_COLORS['snow_storm'][0], 'textAlign': 'center'}),
            html.P("This should not happen - default algorithms should be automatically loaded.",
                   style={'color': NORD_COLORS['aurora'][0], 'textAlign': 'center', 'fontSize': '12px'})
        ])

    algorithm_cards: list = []

    # Separate defaults and user algorithms
    default_algorithms = [alg for alg in algorithms if alg.is_default]
    user_algorithms = [alg for alg in algorithms if not alg.is_default]

    # Add section header for defaults if any exist
    if default_algorithms:
        algorithm_cards.append(
            html.H6("📋 Default Example Algorithms",
                    style={'color': NORD_COLORS['aurora'][3], 'marginTop': '10px', 'marginBottom': '15px'})
        )

    # Process all algorithms
    for alg in default_algorithms + user_algorithms:
        if alg == default_algorithms[0] and user_algorithms:
            # Add section header for user algorithms
            algorithm_cards.append(
                html.H6("👤 Your Uploaded Algorithms",
                        style={'color': NORD_COLORS['frost'][2], 'marginTop': '20px', 'marginBottom': '15px'})
            )

        is_selected: bool = alg.id == selected_id

        status_color: str = {
            'uploaded': NORD_COLORS['frost'][2],
            'tested': NORD_COLORS['aurora'][3],
            'error': NORD_COLORS['aurora'][0]
        }.get(getattr(alg, 'status', 'uploaded'), NORD_COLORS['snow_storm'][1])

        # Different styling for defaults vs user algorithms
        if alg.is_default:
            card_border = f"2px solid {NORD_COLORS['aurora'][3]}"
            card_bg = NORD_COLORS['polar_night'][1]
            default_badge = dbc.Badge("Default", color="success", className="me-2",
                                      style={'fontSize': '10px'})
        else:
            card_border = f"2px solid {NORD_COLORS['frost'][2]}"
            card_bg = NORD_COLORS['polar_night'][1]
            default_badge = None

        card_content: list
        card_style: dict

        if is_selected:
            # --- Style and content for a SELECTED card ---
            card_style = {'marginBottom': '10px', 'border': f"3px solid {NORD_COLORS['frost'][2]}",
                          'backgroundColor': card_bg, 'color': NORD_COLORS['snow_storm'][0]}
            card_content = [
                html.Div([
                    default_badge,
                    html.H6(getattr(alg, 'algorithm_name', 'Unknown Algorithm'),
                            style={'color': NORD_COLORS['snow_storm'][2], 'display': 'inline'})
                ]),
                html.Hr(style={'backgroundColor': NORD_COLORS['aurora'][2], 'height': '3px', 'border': 'none'}),
                html.P([html.Strong("File: "), getattr(alg, 'filename', 'N/A')]),
                html.P([html.Strong("Status: "),
                        html.Span(getattr(alg, 'status', 'unknown').title(), style={'color': status_color})]),
                dbc.Alert("✓ Algorithm selected for testing", color="success",
                          style={'marginTop': '15px', 'padding': '0.5rem 1rem'}),
                dbc.Button(
                    "❌ Deselect",
                    id={'type': 'deselect-algorithm-btn', 'index': alg.id},
                    color="warning",
                    size="sm",
                    outline=True,
                    className="mt-2 w-100"
                )
            ]
        else:
            # --- Style and content for an UNSELECTED card ---
            card_style = {'marginBottom': '10px', 'border': card_border,
                          'backgroundColor': card_bg, 'color': NORD_COLORS['snow_storm'][0]}
            card_content = [
                html.Div([
                    default_badge,
                    html.H6(getattr(alg, 'algorithm_name', 'Unknown Algorithm'),
                            style={'color': NORD_COLORS['snow_storm'][2], 'display': 'inline'})
                ]),
                html.Hr(style={'backgroundColor': NORD_COLORS['aurora'][2], 'height': '3px', 'border': 'none'}),
                html.P([html.Strong("File: "), getattr(alg, 'filename', 'N/A')]),
                html.P([html.Strong("Status: "),
                        html.Span(getattr(alg, 'status', 'unknown').title(), style={'color': status_color})])
            ]

            # Add description for default algorithms
            if alg.is_default and alg.description:
                card_content.append(
                    html.P(alg.description[:100] + "..." if len(alg.description) > 100 else alg.description,
                           style={'fontSize': '12px', 'color': NORD_COLORS['snow_storm'][0], 'fontStyle': 'italic'})
                )

            card_content.append(
                dbc.Button(
                    "Select for Testing",
                    id={'type': 'select-algorithm-btn', 'index': alg.id},
                    size="sm", color="primary",
                    style={'backgroundColor': NORD_COLORS['polar_night'][0], 'borderColor': NORD_COLORS['frost'][2],
                           'width': '100%'}
                )
            )

        algorithm_cards.append(dbc.Card(dbc.CardBody(card_content), style=card_style))

    return html.Div(algorithm_cards)


@app.callback(
    Output('selected-algorithm-store', 'children'),
    [Input({'type': 'select-algorithm-btn', 'index': ALL}, 'n_clicks'),
     Input({'type': 'deselect-algorithm-btn', 'index': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_selection_state(select_clicks: list, deselect_clicks: list) -> str:
    """
    Handles both selecting and deselecting an algorithm.
    It updates the central selected-algorithm-store which triggers the UI refresh.
    """
    triggered_context: dict = callback_context.triggered_id
    if not triggered_context:
        return dash.no_update

    button_type: str = triggered_context.get('type')

    if button_type == 'select-algorithm-btn':
        algorithm_id: int = triggered_context.get('index')
        return str(algorithm_id)

    if button_type == 'deselect-algorithm-btn':
        return ""  # Return empty string to clear the selection

    return dash.no_update


# Test button state callback
@app.callback(
    Output('run-algorithm-test-btn', 'disabled'),
    [Input('test-stream-selector', 'value'),
     Input('selected-algorithm-store', 'children')],
    prevent_initial_call=False
)
def update_test_button_state(stream_value, selected_algorithm):
    try:
        has_stream = stream_value is not None and stream_value != ""
        has_algorithm = selected_algorithm is not None and selected_algorithm != ""
        return not (has_stream and has_algorithm)
    except Exception:
        return True


# Stream description callback
@app.callback(
    [Output('test-stream-description', 'children'),
     Output('test-stream-description', 'style')],
    [Input('test-stream-selector', 'value')]
)
def update_test_stream_description(stream_id):
    if not stream_id:
        return "", {'display': 'none'}

    try:
        description = stream_manager.get_stream_description(stream_id)
        return html.P(description, style={'color': NORD_COLORS['snow_storm'][0], 'fontSize': '12px'}), {
            'color': NORD_COLORS['snow_storm'][0],
            'fontSize': '12px',
            'marginBottom': '15px',
            'padding': '10px',
            'backgroundColor': NORD_COLORS['polar_night'][2],
            'borderRadius': '4px',
            'display': 'block'
        }
    except Exception as e:
        return f"Error loading stream description: {str(e)}", {'display': 'block'}


# Run algorithm test (unchanged)
@app.callback(
    [Output('test-results-plot', 'figure'),
     Output('test-results-plot', 'style'),
     Output('test-results-placeholder', 'style'),
     Output('test-metrics-display', 'children'),
     Output('test-results-store', 'children')],
    [Input('run-algorithm-test-btn', 'n_clicks')],
    [State('selected-algorithm-store', 'children'),
     State('test-stream-selector', 'value'),
     State('test-num-cases-slider', 'value'),
     State('user-token-store', 'children')],
    prevent_initial_call=True
)
def run_algorithm_test(n_clicks, selected_algorithm_id, stream_id, num_cases, token):
    if not n_clicks or not selected_algorithm_id or not stream_id or not token:
        return {}, {'display': 'none'}, {'display': 'block'}, "", ""

    try:
        algorithm_id = int(selected_algorithm_id)

        # Show loading state
        loading_msg = dbc.Alert("🔄 Executing algorithm... This may take a few minutes.",
                                color="info", style={'marginTop': '15px'})

        # Execute algorithm using the real executor
        result = algorithm_executor.execute_algorithm_test(
            algorithm_id, token, stream_id, num_cases
        )

        if not result.success:
            error_msg = dbc.Alert(f"❌ Test failed: {result.error_message}",
                                  color="danger", style={'marginTop': '15px'})
            return {}, {'display': 'none'}, {'display': 'block'}, error_msg, ""

        # Get algorithm info for display
        algorithm = db_manager.get_algorithm_by_id(algorithm_id, token)
        algorithm_name = algorithm.algorithm_name if algorithm else f"Algorithm {algorithm_id}"

        # Create performance plot
        fig = go.Figure()

        # Add baseline line
        fig.add_trace(go.Scatter(
            x=list(range(len(result.baseline_scores))),
            y=result.baseline_scores,
            mode='lines',
            name='Baseline',
            line=dict(color=NORD_COLORS['snow_storm'][0], dash='dash', width=1),
            opacity=0.7
        ))

        # Add algorithm performance line
        fig.add_trace(go.Scatter(
            x=list(range(len(result.conformance_scores))),
            y=result.conformance_scores,
            mode='lines',
            name=algorithm_name,
            line=dict(color=NORD_COLORS['frost'][2], width=2)
        ))

        # Update layout
        fig.update_layout(
            title=f"Algorithm Performance: {algorithm_name} on {stream_id}",
            xaxis_title="Event Number",
            yaxis_title="Conformance Score",
            template='plotly_dark',
            paper_bgcolor=NORD_COLORS['polar_night'][1],
            plot_bgcolor=NORD_COLORS['polar_night'][1],
            font=dict(color=NORD_COLORS['snow_storm'][2]),
            legend=dict(x=0.02, y=0.98),
            hovermode='x unified'
        )

        # Create metrics display
        metrics = result.metrics
        performance_data = result.performance_data

        metrics_display = html.Div([
            html.H6("📊 Test Results", style={'color': NORD_COLORS['snow_storm'][2], 'marginBottom': '15px'}),

            # Performance metrics row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{metrics['mae']:.3f}", style={'color': NORD_COLORS['aurora'][0], 'margin': 0}),
                            html.P("Mean Absolute Error", style={'margin': 0, 'fontSize': '12px'})
                        ])
                    ], style={'textAlign': 'center', 'backgroundColor': NORD_COLORS['polar_night'][2]})
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{metrics['rmse']:.3f}", style={'color': NORD_COLORS['aurora'][1], 'margin': 0}),
                            html.P("Root Mean Square Error", style={'margin': 0, 'fontSize': '12px'})
                        ])
                    ], style={'textAlign': 'center', 'backgroundColor': NORD_COLORS['polar_night'][2]})
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{metrics['accuracy']:.1%}",
                                    style={'color': NORD_COLORS['aurora'][3], 'margin': 0}),
                            html.P("Accuracy (±0.1)", style={'margin': 0, 'fontSize': '12px'})
                        ])
                    ], style={'textAlign': 'center', 'backgroundColor': NORD_COLORS['polar_night'][2]})
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{metrics['global_errors']}",
                                    style={'color': NORD_COLORS['frost'][2], 'margin': 0}),
                            html.P("Global Errors", style={'margin': 0, 'fontSize': '12px'})
                        ])
                    ], style={'textAlign': 'center', 'backgroundColor': NORD_COLORS['polar_night'][2]})
                ], width=3)
            ], className="mb-3"),

            # Performance statistics row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("⚡ Performance", style={'color': NORD_COLORS['snow_storm'][2]}),
                            html.P(f"Execution Time: {result.execution_time:.2f}s",
                                   style={'color': NORD_COLORS['snow_storm'][1], 'margin': '5px 0'}),
                            html.P(f"Memory Usage: {result.memory_usage_mb:.1f}MB",
                                   style={'color': NORD_COLORS['snow_storm'][1], 'margin': '5px 0'}),
                            html.P(
                                f"Events/sec: {num_cases / result.execution_time:.1f}" if result.execution_time > 0 else "Events/sec: N/A",
                                style={'color': NORD_COLORS['snow_storm'][1], 'margin': '5px 0'}),
                        ])
                    ], style={'backgroundColor': NORD_COLORS['polar_night'][2]})
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("📈 Quality", style={'color': NORD_COLORS['snow_storm'][2]}),
                            html.P(f"Correlation: {metrics['correlation']:.3f}",
                                   style={'color': NORD_COLORS['snow_storm'][1], 'margin': '5px 0'}),
                            html.P(f"Mean Score: {metrics['mean_conformance']:.3f}",
                                   style={'color': NORD_COLORS['snow_storm'][1], 'margin': '5px 0'}),
                            html.P(f"Score Range: {metrics['min_conformance']:.3f} - {metrics['max_conformance']:.3f}",
                                   style={'color': NORD_COLORS['snow_storm'][1], 'margin': '5px 0'}),
                        ])
                    ], style={'backgroundColor': NORD_COLORS['polar_night'][2]})
                ], width=6)
            ], className="mb-3"),

            dbc.Alert(f"✅ Test completed successfully on {num_cases} cases",
                      color="success", style={'marginTop': '15px'})
        ])

        # Update algorithm test results in database
        test_results = {
            'stream_id': stream_id,
            'num_cases': num_cases,
            'metrics': metrics,
            'execution_time': result.execution_time,
            'memory_usage_mb': result.memory_usage_mb,
            'timestamp': datetime.now().isoformat()
        }

        # Store results in algorithm record
        try:
            # Update algorithm with test results
            db_manager.update_algorithm_test_results(algorithm_id, token, test_results)
            logger.info(f"Test results saved for algorithm {algorithm_id}")
        except Exception as e:
            logger.warning(f"Failed to save test results: {e}")
            # Continue anyway - test was successful even if saving failed

        return (
            fig,
            {'height': '350px', 'display': 'block'},
            {'display': 'none'},
            metrics_display,
            json.dumps(test_results)
        )

    except Exception as e:
        error_msg = f"Test execution failed: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")

        error_display = dbc.Alert(error_msg, color="danger")
        return {}, {'display': 'none'}, {'display': 'block'}, error_display, ""


# Enhanced display user algorithms callback with default algorithm support
@app.callback(
    Output('user-algorithms-list', 'children'),
    [Input('main-tabs', 'active_tab'),
     Input('user-token-store', 'children'),
     Input('algorithm-refresh-trigger', 'children')],
    prevent_initial_call=False
)
def display_user_algorithms(active_tab, token, refresh_trigger):
    if active_tab != "my-algorithms":
        return html.Div()

    if not token:
        return html.P("Loading...", style={'color': NORD_COLORS['snow_storm'][0]})

    try:
        algorithms = db_manager.get_user_algorithms(token)
    except Exception as e:
        return dbc.Alert(f"Error loading algorithms: {str(e)}", color="danger")

    if not algorithms:
        return html.Div([
            html.P("No algorithms available.",
                   style={'color': NORD_COLORS['snow_storm'][0], 'textAlign': 'center', 'padding': '50px'}),
            html.P("This should not happen - default algorithms should be loaded automatically.",
                   style={'color': NORD_COLORS['aurora'][0], 'textAlign': 'center', 'fontSize': '12px'})
        ])

    # Separate defaults and user algorithms
    default_algorithms = [alg for alg in algorithms if alg.is_default]
    user_algorithms = [alg for alg in algorithms if not alg.is_default]

    algorithm_cards = []

    # Add default algorithms section
    if default_algorithms:
        algorithm_cards.append(
            html.H5("📋 Default Example Algorithms",
                    style={'color': NORD_COLORS['aurora'][3], 'marginBottom': '20px', 'marginTop': '10px'})
        )

        for i, alg in enumerate(default_algorithms):
            algorithm_cards.append(create_algorithm_card(alg, i, is_default=True))

    # Add user algorithms section
    if user_algorithms:
        algorithm_cards.append(
            html.H5("👤 Your Uploaded Algorithms",
                    style={'color': NORD_COLORS['frost'][2], 'marginBottom': '20px', 'marginTop': '30px'})
        )

        for i, alg in enumerate(user_algorithms):
            algorithm_cards.append(create_algorithm_card(alg, i + len(default_algorithms), is_default=False))
    elif not user_algorithms and default_algorithms:
        algorithm_cards.append(
            html.Div([
                html.H5("👤 Your Uploaded Algorithms",
                        style={'color': NORD_COLORS['frost'][2], 'marginBottom': '15px', 'marginTop': '30px'}),
                html.P(
                    "No custom algorithms uploaded yet. Try testing the default algorithms above, then upload your own!",
                    style={'color': NORD_COLORS['snow_storm'][0], 'textAlign': 'center', 'fontStyle': 'italic'})
            ])
        )

    return html.Div(algorithm_cards)


def create_algorithm_card(alg, index, is_default=False):
    """Create an algorithm card with appropriate styling and functionality."""
    try:
        status_color = {
            'uploaded': NORD_COLORS['frost'][2],
            'tested': NORD_COLORS['aurora'][3],
            'error': NORD_COLORS['aurora'][0]
        }.get(getattr(alg, 'status', 'uploaded'), NORD_COLORS['snow_storm'][1])

        # Determine if algorithm has test results
        has_test_results = (hasattr(alg, 'test_results') and alg.test_results and
                            isinstance(alg.test_results, dict))

        # Different styling for default vs user algorithms
        if is_default:
            card_style = {
                'marginBottom': '15px',
                'border': f'2px solid {NORD_COLORS["aurora"][3]}',
                'backgroundColor': NORD_COLORS['polar_night'][1]
            }
            badge = dbc.Badge("Default Example", color="success", className="mb-2")
        else:
            card_style = {
                'marginBottom': '15px',
                'border': f'2px solid {NORD_COLORS["frost"][2]}',
                'backgroundColor': NORD_COLORS['polar_night'][1]
            }
            badge = dbc.Badge("Your Algorithm", color="primary", className="mb-2")

        card_body_content = [
            badge,
            dbc.Row([
                dbc.Col([
                    html.P([html.Strong("File: "), getattr(alg, 'filename', 'N/A')]),
                    html.P([html.Strong("Uploaded: "),
                            getattr(alg, 'upload_time', 'Unknown').strftime('%Y-%m-%d %H:%M')
                            if hasattr(alg, 'upload_time') and alg.upload_time else 'Unknown']),
                    html.P([
                        html.Strong("Status: "),
                        html.Span(getattr(alg, 'status', 'unknown').title(), style={'color': status_color})
                    ]),
                    html.P([
                        html.Strong("Tests Run: "),
                        str(getattr(alg, 'test_count', 0))
                    ])
                ], width=8),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("🧪 Test",
                                   id={'type': 'test-algorithm-btn', 'index': getattr(alg, 'id', index)},
                                   size="sm", color="success",
                                   style={'marginBottom': '5px', 'width': '100%'}),
                        # Only show delete button for non-default algorithms
                        dbc.Button("🗑️ Delete",
                                   id={'type': 'delete-algorithm-btn', 'index': getattr(alg, 'id', index)},
                                   size="sm", color="danger",
                                   style={'width': '100%'},
                                   disabled=is_default) if not is_default else html.Div()
                    ], vertical=True, style={'width': '100%'})
                ], width=4)
            ])
        ]

        # Add description for algorithms that have one
        if hasattr(alg, 'description') and alg.description:
            description_text = alg.description
            if len(description_text) > 150:
                description_text = description_text[:150] + "..."

            card_body_content.insert(-1, html.P(
                description_text,
                style={'color': NORD_COLORS['snow_storm'][0], 'fontSize': '12px',
                       'fontStyle': 'italic', 'marginTop': '10px'}
            ))

        # Add test results if available
        if has_test_results:
            metrics = alg.test_results.get('metrics', {})
            execution_time = alg.test_results.get('execution_time', 0)

            # Calculate composite score
            composite_score = algorithm_executor._calculate_composite_score(metrics)

            test_results_section = html.Div([
                html.Hr(style={'borderColor': NORD_COLORS['polar_night'][3], 'margin': '10px 0'}),
                html.H6("📊 Latest Test Results",
                        style={'color': NORD_COLORS['snow_storm'][2], 'marginBottom': '10px'}),
                dbc.Row([
                    dbc.Col([
                        html.Small(f"Accuracy: {metrics.get('accuracy', 0):.1%}",
                                   style={'color': NORD_COLORS['aurora'][3]})
                    ], width=4),
                    dbc.Col([
                        html.Small(f"MAE: {metrics.get('mae', 0):.3f}",
                                   style={'color': NORD_COLORS['aurora'][1]})
                    ], width=4),
                    dbc.Col([
                        html.Small(f"Score: {composite_score:.3f}",
                                   style={'color': NORD_COLORS['frost'][2], 'fontWeight': 'bold'})
                    ], width=4)
                ]),
                html.Small(f"Execution: {execution_time:.2f}s",
                           style={'color': NORD_COLORS['snow_storm'][0]})
            ])
            card_body_content.append(test_results_section)

        card = dbc.Card([
            dbc.CardHeader([
                html.H6(getattr(alg, 'algorithm_name', 'Unknown Algorithm'),
                        style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0})
            ]),
            dbc.CardBody(card_body_content)
        ], style=card_style)

        return card

    except Exception as e:
        error_card = dbc.Card([
            dbc.CardBody([
                html.P(f"Error displaying algorithm: {str(e)}",
                       style={'color': NORD_COLORS['aurora'][0]})
            ])
        ], style={'marginBottom': '15px'})
        return error_card


@app.callback(
    [Output('submit-to-challenge-btn', 'disabled'),
     Output('submit-to-challenge-btn', 'children')],
    [Input('main-tabs', 'active_tab'),
     Input('user-token-store', 'children'),
     Input('algorithm-refresh-trigger', 'children')],
    prevent_initial_call=False
)
def update_submit_button_state(active_tab, token, refresh_trigger):
    if active_tab != "my-algorithms" or not token:
        return True, "🏆 Submit Best Algorithm to Challenge"

    try:
        # Get best algorithm for user
        best_algorithm = algorithm_executor.get_best_algorithm_for_user(token)

        if best_algorithm:
            return False, f"🏆 Submit '{best_algorithm['name']}' (Score: {best_algorithm['score']:.3f})"
        else:
            return True, "🏆 No Tested Algorithms Available"

    except Exception as e:
        logger.error(f"Error checking best algorithm: {e}")
        return True, "🏆 Submit Best Algorithm to Challenge"


# Enhanced navigation callbacks
@app.callback(
    Output('main-tabs', 'active_tab'),
    [Input('goto-upload-btn', 'n_clicks'),
     Input('goto-upload-from-myalgs-btn', 'n_clicks'),
     Input({'type': 'test-algorithm-btn', 'index': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def handle_navigation(upload_clicks, upload_from_myalgs_clicks, test_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_id = ctx.triggered[0]['prop_id']

    if 'goto-upload-btn' in trigger_id or 'goto-upload-from-myalgs-btn' in trigger_id:
        return "upload"
    elif 'test-algorithm-btn' in trigger_id:
        return "test"

    return dash.no_update


# Challenge submission callback (unchanged)
@app.callback(
    [Output('submit-to-challenge-btn', 'color'),
     Output('submit-to-challenge-btn', 'children', allow_duplicate=True)],
    [Input('submit-to-challenge-btn', 'n_clicks')],
    [State('user-token-store', 'children')],
    prevent_initial_call=True
)
def handle_challenge_submission(n_clicks, token):
    if not n_clicks or not token:
        return "primary", "🏆 Submit Best Algorithm to Challenge"

    try:
        # Get user info and best algorithm
        user_stats = db_manager.get_user_statistics(token)
        best_algorithm_info = algorithm_executor.get_best_algorithm_for_user(token)

        if not best_algorithm_info:
            return "danger", "❌ No Algorithm Available"

        # Get full algorithm details
        algorithm = db_manager.get_algorithm_by_id(best_algorithm_info['id'], token)
        if not algorithm:
            return "danger", "❌ Algorithm Not Found"

        if LEADERBOARD_AVAILABLE and submission_manager:
            # Submit to leaderboard using existing submission_manager
            submission_id = submission_manager.submit_algorithm(
                team_name=user_stats.get('email', 'Anonymous'),  # Use email as team name
                email=user_stats.get('email', ''),
                algorithm_name=algorithm.algorithm_name,
                description=algorithm.description or "No description provided",
                file_path=algorithm.file_path,
                libraries=algorithm.required_libraries or []
            )

            # Start async evaluation if available
            if hasattr(submission_manager, 'evaluate_submission_async'):
                success = submission_manager.evaluate_submission_async(submission_id)
                if success:
                    return "success", "✅ Submitted Successfully!"
                else:
                    return "warning", "⚠️ Submitted (Evaluation Pending)"
            else:
                return "success", "✅ Submitted to Challenge!"
        else:
            # Leaderboard not available - just mark as submitted locally
            return "info", "ℹ️ Submitted (Local Mode)"

    except Exception as e:
        logger.error(f"Challenge submission failed: {e}")
        return "danger", f"❌ Submission Failed"


# Delete algorithm using pattern matching callback (enhanced to prevent default deletion)
@app.callback(
    Output('algorithm-refresh-trigger', 'children'),
    [Input({'type': 'delete-algorithm-btn', 'index': ALL}, 'n_clicks')],
    [State('user-token-store', 'children')],
    prevent_initial_call=True
)
def handle_algorithm_deletion(n_clicks_list, token):
    if not n_clicks_list or not any(n_clicks_list):
        return dash.no_update

    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update

    try:
        # Extract algorithm ID from the triggered component
        button_info = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        algorithm_id = button_info['index']

        # The database manager will prevent deletion of default algorithms
        success = user_db.delete_algorithm(algorithm_id, token)

        if success:
            import time
            return str(int(time.time()))
        else:
            return dash.no_update

    except Exception as e:
        print(f"Error deleting algorithm: {e}")
        return dash.no_update


if __name__ == '__main__':
    PORT: int = 8050
    print("🚀 Starting AVOCADO Streaming Process Mining Challenge..")
    print("=" * 60)
    print(f"📍 Open your browser to: http://localhost:{PORT}")
    print("💡 All users now start with default example algorithms!")
    print("=" * 60)

    try:
        app.run_server(debug=False, host='0.0.0.0', port=PORT)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Application error: {e}")
        print("Check the logs above for details.")