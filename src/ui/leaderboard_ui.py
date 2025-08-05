import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


def create_leaderboard_layout(nord_colors):
    """Create the main leaderboard layout."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🏆 Challenge Leaderboard",
                        className="text-center mb-4",
                        style={'color': nord_colors['snow_storm'][2], 'font-weight': 'bold'})
            ])
        ], className="mb-4"),

        # Stats Cards Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0", id="total-submissions-stat",
                                style={'color': nord_colors['frost'][2], 'margin': 0}),
                        html.P("Total Submissions",
                               style={'color': nord_colors['snow_storm'][1], 'margin': 0})
                    ])
                ], style={'background-color': nord_colors['polar_night'][1], 'border': 'none', 'text-align': 'center'})
            ], width=2),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0", id="unique-teams-stat",
                                style={'color': nord_colors['aurora'][3], 'margin': 0}),
                        html.P("Teams Participating",
                               style={'color': nord_colors['snow_storm'][1], 'margin': 0})
                    ])
                ], style={'background-color': nord_colors['polar_night'][1], 'border': 'none', 'text-align': 'center'})
            ], width=2),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0", id="completed-submissions-stat",
                                style={'color': nord_colors['aurora'][1], 'margin': 0}),
                        html.P("Completed Evaluations",
                               style={'color': nord_colors['snow_storm'][1], 'margin': 0})
                    ])
                ], style={'background-color': nord_colors['polar_night'][1], 'border': 'none', 'text-align': 'center'})
            ], width=2),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0%", id="completion-rate-stat",
                                style={'color': nord_colors['aurora'][2], 'margin': 0}),
                        html.P("Success Rate",
                               style={'color': nord_colors['snow_storm'][1], 'margin': 0})
                    ])
                ], style={'background-color': nord_colors['polar_night'][1], 'border': 'none', 'text-align': 'center'})
            ], width=2),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0", id="recent-submissions-stat",
                                style={'color': nord_colors['frost'][1], 'margin': 0}),
                        html.P("This Week",
                               style={'color': nord_colors['snow_storm'][1], 'margin': 0})
                    ])
                ], style={'background-color': nord_colors['polar_night'][1], 'border': 'none', 'text-align': 'center'})
            ], width=2),

            dbc.Col([
                dbc.Button(
                    "🔄 Refresh",
                    id="refresh-leaderboard-btn",
                    color="primary",
                    style={
                        'background-color': nord_colors['frost'][2],
                        'border-color': nord_colors['frost'][2],
                        'width': '100%'
                    }
                )
            ], width=2)
        ], className="mb-4"),

        # Main Content Row
        dbc.Row([
            # Leaderboard Table
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🏆 Top Submissions",
                                style={'color': nord_colors['snow_storm'][2], 'margin': 0})
                    ], style={'background-color': nord_colors['polar_night'][2]}),
                    dbc.CardBody([
                        html.Div(id="leaderboard-table")
                    ], style={'background-color': nord_colors['polar_night'][1], 'padding': 0})
                ], style={'background-color': nord_colors['polar_night'][1], 'border': 'none'})
            ], width=8),

            # Recent Activity
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("📈 Performance Distribution",
                                style={'color': nord_colors['snow_storm'][2], 'margin': 0})
                    ], style={'background-color': nord_colors['polar_night'][2]}),
                    dbc.CardBody([
                        dcc.Graph(
                            id="performance-distribution-chart",
                            style={'height': '300px'}
                        )
                    ], style={'background-color': nord_colors['polar_night'][1]})
                ], style={'background-color': nord_colors['polar_night'][1], 'border': 'none',
                          'margin-bottom': '20px'}),

                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🕒 Recent Activity",
                                style={'color': nord_colors['snow_storm'][2], 'margin': 0})
                    ], style={'background-color': nord_colors['polar_night'][2]}),
                    dbc.CardBody([
                        html.Div(id="recent-activity-list")
                    ], style={'background-color': nord_colors['polar_night'][1]})
                ], style={'background-color': nord_colors['polar_night'][1], 'border': 'none'})
            ], width=4)
        ]),

        # Detailed Results Modal
        dbc.Modal([
            dbc.ModalHeader([
                dbc.ModalTitle("Submission Details", id="submission-modal-title",
                               style={'color': nord_colors['snow_storm'][2]})
            ], style={'background-color': nord_colors['polar_night'][1]}),
            dbc.ModalBody([
                html.Div(id="submission-details-content")
            ], style={'background-color': nord_colors['polar_night'][1]}),
            dbc.ModalFooter([
                dbc.Button("Close", id="close-submission-modal", color="secondary",
                           style={'background-color': nord_colors['polar_night'][3]})
            ], style={'background-color': nord_colors['polar_night'][1]})
        ], id="submission-details-modal", size="xl", is_open=False),

        # Auto-refresh interval
        dcc.Interval(
            id='leaderboard-refresh-interval',
            interval=30 * 1000,  # 30 seconds
            n_intervals=0
        )

    ], fluid=True, style={
        'background-color': nord_colors['polar_night'][0],
        'min-height': '100vh',
        'padding': '20px'
    })


def create_leaderboard_table(leaderboard_data: List[Dict], nord_colors) -> html.Div:
    """Create the leaderboard table component with fixed score display."""
    if not leaderboard_data:
        return html.Div([
            html.P("No submissions available yet.",
                   style={'text-align': 'center', 'color': nord_colors['snow_storm'][0], 'padding': '50px'})
        ])

    # Create table rows
    table_rows = []

    # Header row - Updated to match new score structure
    header_row = html.Tr([
        html.Th("Rank", style={'color': nord_colors['snow_storm'][2], 'text-align': 'center', 'width': '60px'}),
        html.Th("Team", style={'color': nord_colors['snow_storm'][2], 'width': '180px'}),
        html.Th("Algorithm", style={'color': nord_colors['snow_storm'][2], 'width': '180px'}),
        html.Th("Score", style={'color': nord_colors['snow_storm'][2], 'text-align': 'center', 'width': '80px'}),
        html.Th("Accuracy", style={'color': nord_colors['snow_storm'][2], 'text-align': 'center', 'width': '70px'}),
        html.Th("MAE", style={'color': nord_colors['snow_storm'][2], 'text-align': 'center', 'width': '70px'}),
        html.Th("RMSE", style={'color': nord_colors['snow_storm'][2], 'text-align': 'center', 'width': '70px'}),
        html.Th("Latency", style={'color': nord_colors['snow_storm'][2], 'text-align': 'center', 'width': '70px'}),
        html.Th("Robust", style={'color': nord_colors['snow_storm'][2], 'text-align': 'center', 'width': '70px'}),
        html.Th("Submitted", style={'color': nord_colors['snow_storm'][2], 'text-align': 'center', 'width': '100px'}),
        html.Th("", style={'width': '60px'})  # Actions column
    ], style={'background-color': nord_colors['polar_night'][2]})

    table_rows.append(header_row)

    # Data rows
    for entry in leaderboard_data:
        rank = entry['rank']

        # Medal for top 3
        if rank == 1:
            rank_display = "🥇"
        elif rank == 2:
            rank_display = "🥈"
        elif rank == 3:
            rank_display = "🥉"
        else:
            rank_display = f"#{rank}"

        # Color based on rank
        if rank <= 3:
            row_bg = f"{nord_colors['aurora'][2]}15"
        elif rank <= 10:
            row_bg = f"{nord_colors['frost'][2]}10"
        else:
            row_bg = nord_colors['polar_night'][1]

        row = html.Tr([
            html.Td(rank_display,
                    style={'text-align': 'center', 'font-weight': 'bold', 'color': nord_colors['snow_storm'][2]}),
            html.Td(entry['team_name'], style={'color': nord_colors['snow_storm'][2]}),
            html.Td(entry['algorithm_name'], style={'color': nord_colors['snow_storm'][1]}),
            html.Td(f"{entry['composite_score']:.3f}",
                    style={'text-align': 'center', 'font-weight': 'bold', 'color': nord_colors['frost'][2]}),
            html.Td(f"{entry['accuracy_score']:.2f}",
                    style={'text-align': 'center', 'color': nord_colors['aurora'][3]}),
            html.Td(f"{entry['mae_score']:.2f}",
                    style={'text-align': 'center', 'color': nord_colors['aurora'][1]}),
            html.Td(f"{entry['rmse_score']:.2f}",
                    style={'text-align': 'center', 'color': nord_colors['aurora'][1]}),
            html.Td(f"{entry.get('latency_score', 0.0):.2f}",  # Fixed: use latency_score instead of speed_score
                    style={'text-align': 'center', 'color': nord_colors['aurora'][2]}),
            html.Td(f"{entry['robustness_score']:.2f}",
                    style={'text-align': 'center', 'color': nord_colors['aurora'][3]}),
            html.Td(entry['submission_time'].strftime('%m/%d %H:%M'),
                    style={'text-align': 'center', 'color': nord_colors['snow_storm'][0], 'font-size': '11px'}),
            html.Td([
                dbc.Button("📊",
                          id={'type': 'view-details-btn', 'submission_id': entry['submission_id']},
                          size="sm",
                          color="outline-primary",
                          style={'border-color': nord_colors['frost'][2], 'color': nord_colors['frost'][2]})
            ], style={'text-align': 'center'})
        ], style={'background-color': row_bg}, id=f"leaderboard-row-{entry['submission_id']}")

        table_rows.append(row)

    return html.Div([
        html.Table(table_rows, style={
            'width': '100%',
            'border-collapse': 'collapse'
        }, className="leaderboard-table")
    ], style={'overflow-x': 'auto'})


def create_performance_chart(leaderboard_data: List[Dict], nord_colors) -> go.Figure:
    """Create performance distribution chart."""
    if not leaderboard_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(color=nord_colors['snow_storm'][0], size=14)
        )
    else:
        # Extract scores
        composite_scores = [entry['composite_score'] for entry in leaderboard_data]
        accuracy_scores = [entry['accuracy_score'] for entry in leaderboard_data]

        fig = go.Figure()

        # Add histogram for composite scores
        fig.add_trace(go.Histogram(
            x=composite_scores,
            name='Composite Score',
            nbinsx=10,
            marker_color=nord_colors['frost'][2],
            opacity=0.7
        ))

        # Add box plot for accuracy
        fig.add_trace(go.Box(
            y=accuracy_scores,
            name='Accuracy Distribution',
            marker_color=nord_colors['aurora'][3],
            line_color=nord_colors['aurora'][3]
        ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=nord_colors['polar_night'][1],
        plot_bgcolor=nord_colors['polar_night'][1],
        font=dict(color=nord_colors['snow_storm'][2]),
        xaxis_title="Score",
        yaxis_title="Count",
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=40)
    )

    return fig


def create_recent_activity_list(recent_submissions: List[Dict], nord_colors) -> html.Div:
    """Create recent activity list."""
    if not recent_submissions:
        return html.Div([
            html.P("No recent activity",
                   style={'text-align': 'center', 'color': nord_colors['snow_storm'][0], 'padding': '20px'})
        ])

    activity_items = []

    for submission in recent_submissions[:10]:  # Show last 10
        time_ago = get_time_ago(submission['submission_time'])

        status_color = {
            'completed': nord_colors['aurora'][3],
            'pending': nord_colors['aurora'][2],
            'evaluating': nord_colors['frost'][2],
            'failed': nord_colors['aurora'][0]
        }.get(submission['status'], nord_colors['snow_storm'][1])

        status_icon = {
            'completed': '✅',
            'pending': '⏳',
            'evaluating': '🔄',
            'failed': '❌'
        }.get(submission['status'], '📝')

        item = html.Div([
            html.Div([
                html.Span(status_icon, style={'margin-right': '8px'}),
                html.Strong(submission['team_name'], style={'color': nord_colors['snow_storm'][2]}),
                html.Br(),
                html.Small(submission['algorithm_name'], style={'color': nord_colors['snow_storm'][1]}),
                html.Br(),
                html.Small(f"{time_ago} • ", style={'color': nord_colors['snow_storm'][0]}),
                html.Small(submission['status'].title(), style={'color': status_color, 'font-weight': 'bold'})
            ])
        ], style={
            'padding': '10px',
            'border-bottom': f'1px solid {nord_colors["polar_night"][3]}',
            'border-left': f'3px solid {status_color}'
        })

        activity_items.append(item)

    return html.Div(activity_items, style={'max-height': '400px', 'overflow-y': 'auto'})


def create_submission_details_modal_content(submission_data: Dict, nord_colors) -> html.Div:
    """Create detailed submission modal content with fixed score display."""
    if not submission_data:
        return html.Div("No data available")

    # Basic info section
    basic_info = dbc.Card([
        dbc.CardHeader("📝 Submission Information"),
        dbc.CardBody([
            html.P([html.Strong("Team: "), submission_data.get('team_name', 'N/A')]),
            html.P([html.Strong("Algorithm: "), submission_data.get('algorithm_name', 'N/A')]),
            html.P([html.Strong("Submitted: "), submission_data.get('submission_time', 'N/A')]),
            html.P([html.Strong("Status: "), submission_data.get('status', 'N/A')]),
            html.P([html.Strong("Description: ")]),
            html.Div(submission_data.get('description', 'No description provided'),
                     style={'background-color': nord_colors['polar_night'][2], 'padding': '10px',
                            'border-radius': '4px'})
        ])
    ], style={'margin-bottom': '20px'})

    # Performance metrics section - Fixed to use correct score names
    if submission_data.get('leaderboard_scores'):
        scores = submission_data['leaderboard_scores']

        metrics_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{scores.get('composite_score', 0):.3f}",
                               style={'color': nord_colors['frost'][2]}),
                        html.P("Composite Score", style={'margin': 0, 'font-size': '12px'})
                    ])
                ], style={'text-align': 'center'})
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{scores.get('accuracy_score', 0):.2f}",
                               style={'color': nord_colors['aurora'][3]}),
                        html.P("Accuracy", style={'margin': 0, 'font-size': '12px'})
                    ])
                ], style={'text-align': 'center'})
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{scores.get('mae_score', 0):.3f}",
                               style={'color': nord_colors['aurora'][1]}),
                        html.P("MAE Score", style={'margin': 0, 'font-size': '12px'})
                    ])
                ], style={'text-align': 'center'})
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{scores.get('rmse_score', 0):.3f}",
                               style={'color': nord_colors['aurora'][1]}),
                        html.P("RMSE Score", style={'margin': 0, 'font-size': '12px'})
                    ])
                ], style={'text-align': 'center'})
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{scores.get('latency_score', 0):.3f}",
                               style={'color': nord_colors['aurora'][2]}),
                        html.P("Latency Score", style={'margin': 0, 'font-size': '12px'})
                    ])
                ], style={'text-align': 'center'})
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{scores.get('robustness_score', 0):.3f}",
                               style={'color': nord_colors['aurora'][3]}),
                        html.P("Robustness", style={'margin': 0, 'font-size': '12px'})
                    ])
                ], style={'text-align': 'center'})
            ], width=2)
        ], className="mb-4")

        # Score breakdown explanation
        breakdown_info = dbc.Card([
            dbc.CardHeader("📈 Score Breakdown"),
            dbc.CardBody([
                html.P("Final Score Formula:", style={'font-weight': 'bold'}),
                html.P("Score = (0.3 × Accuracy) + (0.25 × MAE) + (0.2 × RMSE) + (0.15 × Latency) + (0.1 × Robustness)"),
                html.Hr(),
                html.P("Score Components:", style={'font-weight': 'bold', 'margin-top': '15px'}),
                html.Ul([
                    html.Li(f"Accuracy: {scores.get('accuracy_score', 0):.3f} × 0.30 = {scores.get('accuracy_score', 0) * 0.30:.3f}"),
                    html.Li(f"MAE: {scores.get('mae_score', 0):.3f} × 0.25 = {scores.get('mae_score', 0) * 0.25:.3f}"),
                    html.Li(f"RMSE: {scores.get('rmse_score', 0):.3f} × 0.20 = {scores.get('rmse_score', 0) * 0.20:.3f}"),
                    html.Li(f"Latency: {scores.get('latency_score', 0):.3f} × 0.15 = {scores.get('latency_score', 0) * 0.15:.3f}"),
                    html.Li(f"Robustness: {scores.get('robustness_score', 0):.3f} × 0.10 = {scores.get('robustness_score', 0) * 0.10:.3f}")
                ]),
                html.P([
                    html.Strong("Total: "),
                    f"{scores.get('composite_score', 0):.3f}"
                ], style={'margin-top': '10px', 'font-size': '16px'})
            ])
        ], style={'margin-bottom': '20px'})

        performance_section = html.Div([
            dbc.Card([
                dbc.CardHeader("📊 Performance Metrics"),
                dbc.CardBody([metrics_cards])
            ], style={'margin-bottom': '20px'}),
            breakdown_info
        ])
    else:
        performance_section = dbc.Card([
            dbc.CardHeader("📊 Performance Metrics"),
            dbc.CardBody([
                html.P("No performance metrics available. The submission may not have completed evaluation.",
                      style={'color': nord_colors['aurora'][0]})
            ])
        ], style={'margin-bottom': '20px'})

    # Evaluation details section
    eval_details = dbc.Card([
        dbc.CardHeader("🔬 Evaluation Details"),
        dbc.CardBody([
            html.P("This submission was evaluated across multiple test streams and case counts."),
            html.P("The scores above represent normalized metrics where higher values indicate better performance."),
            html.Ul([
                html.Li("Accuracy: Percentage of predictions within 0.1 of baseline"),
                html.Li("MAE Score: Normalized Mean Absolute Error (higher = lower error)"),
                html.Li("RMSE Score: Normalized Root Mean Square Error (higher = lower error)"),
                html.Li("Latency Score: Processing speed score (higher = faster)"),
                html.Li("Robustness Score: Error resilience score (higher = fewer major errors)")
            ])
        ])
    ])

    return html.Div([basic_info, performance_section, eval_details])


def get_time_ago(timestamp: datetime) -> str:
    """Get human-readable time ago string."""
    now = datetime.now()
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

    diff = now - timestamp

    if diff.days > 0:
        return f"{diff.days}d ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours}h ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes}m ago"
    else:
        return "Just now"


# Additional utility functions for the leaderboard
def format_score(score: float, metric_type: str = 'default') -> str:
    """Format score for display based on metric type."""
    if metric_type == 'percentage':
        return f"{score:.1%}"
    elif metric_type == 'time':
        if score < 0.001:
            return f"{score * 1000:.1f}μs"
        elif score < 1.0:
            return f"{score * 1000:.1f}ms"
        else:
            return f"{score:.2f}s"
    else:
        return f"{score:.3f}"


def get_rank_badge(rank: int, nord_colors) -> html.Span:
    """Get rank badge with appropriate styling."""
    if rank == 1:
        return html.Span("🥇 #1", style={'color': nord_colors['aurora'][2], 'font-weight': 'bold'})
    elif rank == 2:
        return html.Span("🥈 #2", style={'color': nord_colors['snow_storm'][0], 'font-weight': 'bold'})
    elif rank == 3:
        return html.Span("🥉 #3", style={'color': nord_colors['aurora'][1], 'font-weight': 'bold'})
    elif rank <= 10:
        return html.Span(f"#{rank}", style={'color': nord_colors['frost'][2], 'font-weight': 'bold'})
    else:
        return html.Span(f"#{rank}", style={'color': nord_colors['snow_storm'][1]})


# Export functions for use in main app
__all__ = [
    'create_leaderboard_layout',
    'create_leaderboard_table',
    'create_performance_chart',
    'create_recent_activity_list',
    'create_submission_details_modal_content',
    'get_time_ago',
    'format_score',
    'get_rank_badge'
]