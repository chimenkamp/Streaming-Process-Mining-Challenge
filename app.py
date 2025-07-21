# Fixed version of app.py with proper component management

import inspect
import os
import zipfile
import uuid
import base64

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import sys
import importlib.util
from typing import Dict, List, Any, Optional, Type

from src.ui.leaderboard_system import SubmissionManager
from src.ui.leaderboard_ui import create_leaderboard_layout, create_leaderboard_table, create_performance_chart, \
    create_recent_activity_list

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'ui'))

# Import our custom components and classes
from src.ui.algorithm_base import BaseAlgorithm
from src.ui.styles import get_nord_styles, NORD_COLORS, get_custom_css
from src.ui.stream_manager import StreamManager



LEADERBOARD_AVAILABLE = True


# Initialize Dash app with assets folder
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                assets_folder='assets',
                suppress_callback_exceptions=True)  # This is crucial!
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

# Initialize managers
stream_manager = StreamManager()

if LEADERBOARD_AVAILABLE:
    submission_manager = SubmissionManager("uploaded_algorithms")
    submission_manager.set_stream_manager(stream_manager)
else:
    submission_manager = SubmissionManager("uploaded_algorithms")  # Dummy manager

# Create uploads directory
UPLOAD_DIRECTORY = os.path.join(os.getcwd(), "uploaded_algorithms")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


# File handling functions (same as before)
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


def find_concrete_algorithm_class(exec_globals: Dict[str, Any], base_class: Type[BaseAlgorithm]) -> Optional[
    Type[BaseAlgorithm]]:
    """Scan exec_globals to find a concrete subclass of base_class."""
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


# Create main navigation
def create_navigation():
    """Create main navigation tabs."""
    return dbc.Tabs([
        dbc.Tab(label="🧪 Test & Submit", tab_id="test-submit", active_tab_style={'font-weight': 'bold'}),
        dbc.Tab(label="🏆 Leaderboard", tab_id="leaderboard", active_tab_style={'font-weight': 'bold'}),
        dbc.Tab(label="📊 My Submissions", tab_id="my-submissions", active_tab_style={'font-weight': 'bold'}),
        dbc.Tab(label="📚 Documentation", tab_id="documentation", active_tab_style={'font-weight': 'bold'})
    ], id="main-tabs", active_tab="test-submit", style={'margin-bottom': '20px'})


# Create test & submit layout (existing functionality)
def create_test_submit_layout():
    """Create the test and submit layout."""
    return dbc.Row([
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
                                   style={'font-size': '48px', 'color': NORD_COLORS['snow_storm'][0],
                                          'margin-bottom': '20px'}),
                            html.H5("No Algorithm Uploaded",
                                    style={'color': NORD_COLORS['snow_storm'][1], 'margin-bottom': '10px'}),
                            html.P(
                                "Upload a Python file (.py) or ZIP archive (.zip) containing your algorithm implementation.",
                                style={'color': NORD_COLORS['snow_storm'][0], 'text-align': 'center',
                                       'margin-bottom': '20px'}),
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
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "🧪 Run Test",
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
                        ], width=6),
                        dbc.Col([
                            dbc.Button(
                                "🏆 Submit to Challenge",
                                id="submit-to-challenge-btn",
                                color="primary",
                                disabled=True,
                                style={
                                    'background-color': NORD_COLORS['frost'][2],
                                    'border-color': NORD_COLORS['frost'][2],
                                    'border-radius': '6px',
                                    'width': '100%'
                                }
                            )
                        ], width=6)
                    ])
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
            ], style={**nord_styles['card'], 'margin-bottom': '20px'})
        ], width=6)
    ])


# Create my submissions layout
def create_my_submissions_layout():
    """Create layout for viewing user's submissions."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("📊 My Submissions", style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0})
                    ], style=nord_styles['card_header']),
                    dbc.CardBody([
                        dbc.InputGroup([
                            dbc.Input(
                                id="team-name-filter",
                                placeholder="Enter your team name to view submissions...",
                                style={'background-color': NORD_COLORS['polar_night'][1]}
                            ),
                            dbc.Button("🔍 Search", id="search-submissions-btn", color="primary")
                        ]),
                        html.Hr(),
                        html.Div(id="my-submissions-content")
                    ], style=nord_styles['card_body'])
                ], style=nord_styles['card'])
            ])
        ])
    ])


# Create documentation layout
def create_documentation_layout():
    """Create documentation and help layout."""
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

1. **Implement Your Algorithm**: Create a Python class that inherits from `BaseAlgorithm`
2. **Upload**: Upload your Python file or ZIP archive
3. **Test**: Run tests on different event streams
4. **Submit**: Submit to the challenge leaderboard

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
- `concept:origin`: Origin log (for concept drift streams)

## Evaluation Metrics

- **Accuracy**: Percentage of predictions within 0.1 of baseline
- **MAE**: Mean Absolute Error compared to baseline
- **RMSE**: Root Mean Square Error
- **Speed**: Average processing time per event
- **Robustness**: Number of major errors (>0.3 difference)

## Example Algorithm

See the example algorithm file for a complete implementation.
                        """, style={'color': NORD_COLORS['snow_storm'][2]})
                    ], style=nord_styles['card_body'])
                ], style=nord_styles['card'])
            ])
        ])
    ])


# FIXED: Main Layout with persistent components
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Streaming Process Mining Challenge",
                    className="text-center mb-4",
                    style={'color': NORD_COLORS['snow_storm'][2], 'font-weight': 'bold'})
        ])
    ], className="mb-4"),

    # Navigation
    create_navigation(),

    # Tab Content
    html.Div(id="tab-content"),

    # FIXED: Always-present modals and stores (not dependent on tab)
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

                html.H6("Required Libraries (Optional)",
                        style={'color': NORD_COLORS['snow_storm'][2], 'margin-bottom': '15px'}),
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

    # Submission Confirmation Modal
    dbc.Modal([
        dbc.ModalHeader([
            dbc.ModalTitle("Submit to Challenge", style={'color': NORD_COLORS['snow_storm'][2]})
        ], style={'background-color': NORD_COLORS['polar_night'][1]}),
        dbc.ModalBody([
            dbc.Form([
                dbc.Row([
                    dbc.Label("Team Name:", style={'color': NORD_COLORS['snow_storm'][1]}),
                    dbc.Input(id="submission-team-name", type="text", placeholder="Enter your team name")
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Email:", style={'color': NORD_COLORS['snow_storm'][1]}),
                    dbc.Input(id="submission-email", type="email", placeholder="contact@example.com")
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Algorithm Name:", style={'color': NORD_COLORS['snow_storm'][1]}),
                    dbc.Input(id="submission-algorithm-name", type="text", placeholder="My Algorithm v1.0")
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Description:", style={'color': NORD_COLORS['snow_storm'][1]}),
                    dbc.Textarea(id="submission-description", placeholder="Brief description of your approach...")
                ], className="mb-3")
            ])
        ], style={'background-color': NORD_COLORS['polar_night'][1]}),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="submission-cancel-btn", color="secondary"),
            dbc.Button("Submit to Challenge", id="submission-confirm-btn", color="primary")
        ], style={'background-color': NORD_COLORS['polar_night'][1]})
    ], id="submission-modal", is_open=False),

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

    # FIXED: Persistent data stores
    html.Div(id='session-id', style={'display': 'none'}, children=str(uuid.uuid4())),
    html.Div(id='test-results-store', style={'display': 'none'}),
    html.Div(id='stream-data-store', style={'display': 'none'}),
    html.Div(id='current-position-store', style={'display': 'none'}, children='0'),
    html.Div(id='uploaded-algorithm-store', style={'display': 'none'}),
    html.Div(id='submission-status-store', style={'display': 'none'}),

], fluid=True, style={
    'background-color': NORD_COLORS['polar_night'][0],
    'min-height': '100vh',
    'padding': '20px'
})


# FIXED: Callback to handle tab switching
@app.callback(
    Output("tab-content", "children"),
    [Input("main-tabs", "active_tab")]
)
def render_tab_content(active_tab):
    if active_tab == "test-submit":
        return create_test_submit_layout()
    elif active_tab == "leaderboard":
        return create_leaderboard_layout(NORD_COLORS)
    elif active_tab == "my-submissions":
        return create_my_submissions_layout()
    elif active_tab == "documentation":
        return create_documentation_layout()
    else:
        return html.Div("Select a tab")


# FIXED: Callback to open upload modal - with proper error handling
@app.callback(
    Output("upload-modal", "is_open"),
    [Input("upload-btn", "n_clicks"),
     Input("reupload-btn", "n_clicks"),
     Input("upload-cancel-btn", "n_clicks"),
     Input("upload-confirm-btn", "n_clicks")],
    [State("upload-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_upload_modal(upload_clicks, reupload_clicks, cancel_clicks, confirm_clicks, is_open):
    # Check which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        return False

    # Any button click toggles the modal
    return not is_open



# Leaderboard callbacks
@app.callback(
    [Output("total-submissions-stat", "children"),
     Output("unique-teams-stat", "children"),
     Output("completed-submissions-stat", "children"),
     Output("completion-rate-stat", "children"),
     Output("recent-submissions-stat", "children"),
     Output("leaderboard-table", "children"),
     Output("performance-distribution-chart", "figure"),
     Output("recent-activity-list", "children")],
    [Input("leaderboard-refresh-interval", "n_intervals"),
     Input("refresh-leaderboard-btn", "n_clicks")]
)
def update_leaderboard(n_intervals, refresh_clicks):
    if not LEADERBOARD_AVAILABLE:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=NORD_COLORS['polar_night'][1],
            plot_bgcolor=NORD_COLORS['polar_night'][1],
            font=dict(color=NORD_COLORS['snow_storm'][2]),
            title="Leaderboard not available"
        )
        return "0", "0", "0", "0%", "0", html.Div("Leaderboard not available"), empty_fig, html.Div("No activity")

    try:
        # Get leaderboard data
        leaderboard_data = submission_manager.leaderboard_manager.get_leaderboard(50)
        stats = submission_manager.leaderboard_manager.get_leaderboard_stats()

        # Get recent submissions
        all_submissions = submission_manager.database.get_submissions(20)

        # Update stats
        total_submissions = stats['total_submissions']
        unique_teams = stats['unique_teams']
        completed_submissions = stats['completed_submissions']
        completion_rate = f"{stats['completion_rate']:.1%}"
        recent_submissions = stats['recent_submissions']

        # Create table
        table = create_leaderboard_table(leaderboard_data, NORD_COLORS)

        # Create chart
        chart = create_performance_chart(leaderboard_data, NORD_COLORS)

        # Create activity list
        recent_activity = create_recent_activity_list([
            {
                'team_name': s.team_name,
                'algorithm_name': s.algorithm_name,
                'status': s.status,
                'submission_time': s.submission_time
            }
            for s in all_submissions
        ], NORD_COLORS)

        return (
            str(total_submissions),
            str(unique_teams),
            str(completed_submissions),
            completion_rate,
            str(recent_submissions),
            table,
            chart,
            recent_activity
        )
    except Exception as e:
        print(f"Error updating leaderboard: {e}")
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=NORD_COLORS['polar_night'][1],
            plot_bgcolor=NORD_COLORS['polar_night'][1],
            font=dict(color=NORD_COLORS['snow_storm'][2]),
            title="Error loading leaderboard"
        )
        return "Error", "Error", "Error", "Error", "Error", html.Div("Error loading data"), empty_fig, html.Div("Error")


# My submissions callback
@app.callback(
    Output("my-submissions-content", "children"),
    [Input("search-submissions-btn", "n_clicks")],
    [State("team-name-filter", "value")]
)
def update_my_submissions(n_clicks, team_name):
    if not LEADERBOARD_AVAILABLE:
        return html.P("Submission tracking not available",
                      style={'color': NORD_COLORS['snow_storm'][0], 'text-align': 'center', 'padding': '50px'})

    if not team_name:
        return html.P("Enter your team name to view submissions",
                      style={'color': NORD_COLORS['snow_storm'][0], 'text-align': 'center', 'padding': '50px'})

    try:
        # Get team submissions
        team_submissions = submission_manager.leaderboard_manager.get_team_submissions(team_name)

        if not team_submissions:
            return html.P(f"No submissions found for team '{team_name}'",
                          style={'color': NORD_COLORS['snow_storm'][0], 'text-align': 'center', 'padding': '50px'})

        # Create submission cards
        submission_cards = []
        for submission in team_submissions:
            status_color = {
                'completed': NORD_COLORS['aurora'][3],
                'pending': NORD_COLORS['aurora'][2],
                'evaluating': NORD_COLORS['frost'][2],
                'failed': NORD_COLORS['aurora'][0]
            }.get(submission['status'], NORD_COLORS['snow_storm'][1])

            card = dbc.Card([
                dbc.CardHeader([
                    html.H6(submission['algorithm_name'], style={'color': NORD_COLORS['snow_storm'][2], 'margin': 0})
                ]),
                dbc.CardBody([
                    html.P([
                        html.Strong("Status: "),
                        html.Span(submission['status'].title(), style={'color': status_color})
                    ]),
                    html.P([
                        html.Strong("Submitted: "),
                        submission['submission_time'].strftime('%Y-%m-%d %H:%M')
                    ]),
                    html.P([
                        html.Strong("Score: "),
                        f"{submission['composite_score']:.3f}" if submission['composite_score'] > 0 else "N/A"
                    ])
                ])
            ], style={'margin-bottom': '10px'})

            submission_cards.append(card)

        return html.Div(submission_cards)
    except Exception as e:
        return html.P(f"Error loading submissions: {str(e)}",
                      style={'color': NORD_COLORS['aurora'][0], 'text-align': 'center', 'padding': '50px'})


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

def find_concrete_algorithm_class(exec_globals: Dict[str, Any], base_class: Type[BaseAlgorithm]) -> Optional[
    Type[BaseAlgorithm]]:
    """Scan exec_globals to find a concrete subclass of base_class."""
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


if __name__ == '__main__':
    PORT: int = 8050
    print(f"📍 Open your browser to: http://localhost:{PORT}")
    app.run_server(debug=True, host='0.0.0.0', port=PORT)