# Remaining callbacks and utility functions for the complete enhanced app
import json

from dash import Output, Input, State, html, clientside_callback, callback_context

from app import app, nord_styles, stream_manager, load_algorithm_from_file
from src.ui.algorithm_base import EventStream
from src.ui.styles import NORD_COLORS


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

# Callback for handling submission detail modals in leaderboard
@app.callback(
    [Output("submission-details-modal", "is_open"),
     Output("submission-details-content", "children")],
    [Input({"type": "view-details", "index": ALL}, "n_clicks"),
     Input("close-submission-modal", "n_clicks")],
    [State("submission-details-modal", "is_open")]
)
def handle_submission_details(detail_clicks, close_clicks, is_open):
    ctx = callback_context
    
    if not ctx.triggered:
        return False, html.Div()
    
    if close_clicks:
        return False, html.Div()
    
    # Check if any detail button was clicked
    for click_data in ctx.triggered:
        if click_data['value'] and 'view-details' in click_data['prop_id']:
            # Extract submission ID from the button ID
            import re
            match = re.search(r'view-details-(.+?)"', click_data['prop_id'])
            if match:
                submission_id = match.group(1)
                
                # Get submission details
                submission = submission_manager.database.get_submission(submission_id)
                if submission:
                    content = create_submission_details_modal_content({
                        'team_name': submission.team_name,
                        'algorithm_name': submission.algorithm_name,
                        'submission_time': submission.submission_time.strftime('%Y-%m-%d %H:%M'),
                        'status': submission.status,
                        'description': submission.description,
                        'leaderboard_scores': submission.leaderboard_scores
                    }, NORD_COLORS)
                    
                    return True, content
    
    return is_open, html.Div()

# Add error boundary callback for graceful error handling
@app.callback(
    Output('app-error-display', 'children'),
    [Input('app-error-trigger', 'data')],
    prevent_initial_call=True
)
def display_error(error_data):
    if error_data:
        return dbc.Alert(
            f"Application Error: {error_data.get('message', 'Unknown error occurred')}",
            color="danger",
            dismissable=True,
            style={'margin': '20px'}
        )
    return html.Div()

# Initialize database tables on startup
def initialize_app():
    """Initialize application components."""
    try:
        # Ensure database is initialized
        submission_manager.database._init_database()
        
        # Create upload directories
        os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
        
        print("✅ Application initialized successfully")
        print(f"📁 Upload directory: {UPLOAD_DIRECTORY}")
        print(f"🗃️ Database: {submission_manager.database.db_path}")
        
    except Exception as e:
        print(f"❌ Error initializing application: {e}")

# Call initialization
initialize_app()

# Additional utility functions for background processing
class BackgroundTaskManager:
    """Manages background evaluation tasks."""
    
    def __init__(self):
        self.running_evaluations = {}
        self.completed_evaluations = {}
    
    def start_evaluation(self, submission_id: str):
        """Start evaluation in background thread."""
        if submission_id in self.running_evaluations:
            return False  # Already running
        
        def run_evaluation():
            try:
                self.running_evaluations[submission_id] = 'running'
                success = submission_manager.evaluate_submission_async(submission_id)
                self.completed_evaluations[submission_id] = 'success' if success else 'failed'
                del self.running_evaluations[submission_id]
            except Exception as e:
                self.completed_evaluations[submission_id] = f'error: {str(e)}'
                if submission_id in self.running_evaluations:
                    del self.running_evaluations[submission_id]
        
        eval_thread = threading.Thread(target=run_evaluation)
        eval_thread.daemon = True
        eval_thread.start()
        
        return True
    
    def get_status(self, submission_id: str) -> str:
        """Get evaluation status."""
        if submission_id in self.running_evaluations:
            return 'running'
        elif submission_id in self.completed_evaluations:
            return self.completed_evaluations[submission_id]
        else:
            return 'not_started'

# Global task manager
task_manager = BackgroundTaskManager()

# Enhanced submission callback with background evaluation
@app.callback(
    [Output('submission-status-store', 'children', allow_duplicate=True),
     Output('submit-to-challenge-btn', 'children', allow_duplicate=True)],
    [Input('submission-confirm-btn', 'n_clicks')],
    [State('submission-team-name', 'value'),
     State('submission-email', 'value'),
     State('submission-algorithm-name', 'value'),
     State('submission-description', 'value'),
     State('uploaded-algorithm-store', 'children')],
    prevent_initial_call=True
)
def handle_submission_enhanced(n_clicks, team_name, email, algorithm_name, description, algorithm_data_json):
    if not n_clicks:
        return "", "🏆 Submit to Challenge"
    
    if not all([team_name, email, algorithm_name, description]):
        return "error:Missing required fields", "❌ Please fill all fields"
    
    if not algorithm_data_json:
        return "error:No algorithm uploaded", "❌ No algorithm uploaded"
    
    try:
        algorithm_data = json.loads(algorithm_data_json)
        libraries = algorithm_data.get('libraries', '').split('\n') if algorithm_data.get('libraries') else []
        
        # Submit to challenge
        submission_id = submission_manager.submit_algorithm(
            team_name=team_name,
            email=email,
            algorithm_name=algorithm_name,
            description=description,
            file_path=algorithm_data['file_path'],
            libraries=libraries
        )
        
        # Start background evaluation
        task_manager.start_evaluation(submission_id)
        
        return f"success:{submission_id}", "✅ Submitted! Evaluation in progress..."
        
    except Exception as e:
        return f"error:{str(e)}", "❌ Submission failed"

print("🚀 Enhanced Streaming Process Mining Challenge loaded successfully!")
print("📋 Features available:")
print("   • File upload (.py and .zip)")
print("   • Real-time algorithm testing")
print("   • Automatic leaderboard ranking")
print("   • Background evaluation processing")
print("   • Comprehensive submission management")
print("   • Interactive result visualization")