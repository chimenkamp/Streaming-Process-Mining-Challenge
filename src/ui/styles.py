"""
Nord color scheme and styling definitions for the Streaming Process Mining Challenge platform.

The Nord color palette provides a clean, modern aesthetic with good contrast
for both light and dark themes. This file centralizes all styling to ensure
consistent visual design across the application.
"""

# Nord Color Palette
NORD_COLORS = {
    # Polar Night - Dark colors for backgrounds
    'polar_night': [
        '#2e3440',  # nord0 - darkest
        '#3b4252',  # nord1
        '#434c5e',  # nord2
        '#4c566a'   # nord3 - lightest dark
    ],

    # Snow Storm - Light colors for text and light backgrounds
    'snow_storm': [
        '#d8dee9',  # nord4 - darkest light
        '#e5e9f0',  # nord5
        '#eceff4'   # nord6 - lightest
    ],

    # Frost - Blue colors for accents and highlights
    'frost': [
        '#8fbcbb',  # nord7 - cyan
        '#88c0d0',  # nord8 - light blue
        '#81a1c1',  # nord9 - blue
        '#5e81ac'   # nord10 - dark blue
    ],

    # Aurora - Colorful accents for highlights and status
    'aurora': [
        '#bf616a',  # nord11 - red
        '#d08770',  # nord12 - orange
        '#ebcb8b',  # nord13 - yellow
        '#a3be8c',  # nord14 - green
        '#b48ead'   # nord15 - purple
    ]
}

def get_nord_styles():
    """
    Get a dictionary of common style definitions using Nord colors.

    Returns:
        Dict[str, Dict[str, str]]: Dictionary of style definitions
    """
    return {
        # Main container styles
        'main_container': {
            'background-color': NORD_COLORS['polar_night'][0],
            'min-height': '100vh',
            'padding': '20px',
            'font-family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
        },

        # Card styles
        'card': {
            'background-color': NORD_COLORS['polar_night'][1],
            'border': 'none',
            'border-radius': '12px',
            'box-shadow': f'0 4px 6px rgba(0, 0, 0, 0.1)',
            'margin-bottom': '20px'
        },

        'card_header': {
            'background-color': NORD_COLORS['polar_night'][2],
            'border-bottom': f'1px solid {NORD_COLORS["polar_night"][3]}',
            'border-radius': '12px 12px 0 0',
            'padding': '15px 20px'
        },

        'card_body': {
            'background-color': NORD_COLORS['polar_night'][1],
            'padding': '20px',
            'border-radius': '0 0 12px 12px'
        },

        # Button styles
        'button_primary': {
            'background-color': NORD_COLORS['frost'][2],
            'border-color': NORD_COLORS['frost'][2],
            'color': NORD_COLORS['snow_storm'][2],
            'border-radius': '8px',
            'padding': '10px 20px',
            'font-weight': 'bold',
            'transition': 'all 0.3s ease'
        },

        'button_success': {
            'background-color': NORD_COLORS['aurora'][3],
            'border-color': NORD_COLORS['aurora'][3],
            'color': NORD_COLORS['snow_storm'][2],
            'border-radius': '8px',
            'padding': '10px 20px',
            'font-weight': 'bold'
        },

        'button_warning': {
            'background-color': NORD_COLORS['aurora'][2],
            'border-color': NORD_COLORS['aurora'][2],
            'color': NORD_COLORS['polar_night'][0],
            'border-radius': '8px',
            'padding': '10px 20px',
            'font-weight': 'bold'
        },

        'button_danger': {
            'background-color': NORD_COLORS['aurora'][0],
            'border-color': NORD_COLORS['aurora'][0],
            'color': NORD_COLORS['snow_storm'][2],
            'border-radius': '8px',
            'padding': '10px 20px',
            'font-weight': 'bold'
        },

        # Input styles
        'input_field': {
            'background-color': NORD_COLORS['polar_night'][1],
            'border': f'2px solid {NORD_COLORS["polar_night"][3]}',
            'color': NORD_COLORS['snow_storm'][2],
            'border-radius': '6px',
            'padding': '10px 12px',
            'font-size': '14px'
        },

        'textarea': {
            'background-color': NORD_COLORS['polar_night'][1],
            'border': f'2px solid {NORD_COLORS["polar_night"][3]}',
            'color': NORD_COLORS['snow_storm'][2],
            'border-radius': '6px',
            'padding': '10px 12px',
            'font-size': '14px',
            'resize': 'vertical'
        },

        # Code editor styles
        'code_editor': {
            'background-color': NORD_COLORS['polar_night'][1],
            'border': f'1px solid {NORD_COLORS["polar_night"][2]}',
            'color': NORD_COLORS['snow_storm'][2],
            'border-radius': '8px',
            'padding': '15px',
            'font-family': "'Monaco', 'Consolas', 'Lucida Console', monospace",
            'font-size': '13px',
            'line-height': '1.5'
        },

        # Dropdown styles
        'dropdown': {
            'background-color': NORD_COLORS['polar_night'][2],
            'border': f'1px solid {NORD_COLORS["polar_night"][3]}',
            'color': NORD_COLORS['snow_storm'][2],
            'border-radius': '6px'
        },

        # Text styles
        'heading_primary': {
            'color': NORD_COLORS['snow_storm'][2],
            'font-weight': 'bold',
            'margin-bottom': '20px'
        },

        'heading_secondary': {
            'color': NORD_COLORS['snow_storm'][1],
            'font-weight': '600',
            'margin-bottom': '15px'
        },

        'text_primary': {
            'color': NORD_COLORS['snow_storm'][2]
        },

        'text_secondary': {
            'color': NORD_COLORS['snow_storm'][1]
        },

        'text_muted': {
            'color': NORD_COLORS['snow_storm'][0]
        },

        # Alert styles
        'alert_success': {
            'background-color': f'{NORD_COLORS["aurora"][3]}20',
            'border': f'1px solid {NORD_COLORS["aurora"][3]}',
            'color': NORD_COLORS['aurora'][3],
            'border-radius': '8px',
            'padding': '12px 16px'
        },

        'alert_warning': {
            'background-color': f'{NORD_COLORS["aurora"][2]}20',
            'border': f'1px solid {NORD_COLORS["aurora"][2]}',
            'color': NORD_COLORS['aurora'][2],
            'border-radius': '8px',
            'padding': '12px 16px'
        },

        'alert_danger': {
            'background-color': f'{NORD_COLORS["aurora"][0]}20',
            'border': f'1px solid {NORD_COLORS["aurora"][0]}',
            'color': NORD_COLORS['aurora'][0],
            'border-radius': '8px',
            'padding': '12px 16px'
        },

        'alert_info': {
            'background-color': f'{NORD_COLORS["frost"][2]}20',
            'border': f'1px solid {NORD_COLORS["frost"][2]}',
            'color': NORD_COLORS['frost'][2],
            'border-radius': '8px',
            'padding': '12px 16px'
        },

        # Progress bar styles
        'progress_bar': {
            'background-color': NORD_COLORS['polar_night'][3],
            'border-radius': '10px',
            'height': '8px'
        },

        'progress_fill': {
            'background-color': NORD_COLORS['frost'][2],
            'border-radius': '10px',
            'height': '100%',
            'transition': 'width 0.3s ease'
        },

        # Table styles
        'table': {
            'background-color': NORD_COLORS['polar_night'][1],
            'color': NORD_COLORS['snow_storm'][2],
            'border-radius': '8px',
            'overflow': 'hidden'
        },

        'table_header': {
            'background-color': NORD_COLORS['polar_night'][2],
            'color': NORD_COLORS['snow_storm'][2],
            'font-weight': 'bold',
            'padding': '12px'
        },

        'table_cell': {
            'padding': '10px 12px',
            'border-bottom': f'1px solid {NORD_COLORS["polar_night"][2]}'
        },

        # Navigation styles
        'navbar': {
            'background-color': NORD_COLORS['polar_night'][1],
            'border-bottom': f'1px solid {NORD_COLORS["polar_night"][2]}',
            'padding': '15px 0'
        },

        'nav_link': {
            'color': NORD_COLORS['snow_storm'][1],
            'text-decoration': 'none',
            'padding': '8px 16px',
            'border-radius': '6px',
            'transition': 'all 0.3s ease'
        },

        'nav_link_active': {
            'color': NORD_COLORS['frost'][2],
            'background-color': f'{NORD_COLORS["frost"][2]}20'
        },

        # Badge styles
        'badge_primary': {
            'background-color': NORD_COLORS['frost'][2],
            'color': NORD_COLORS['snow_storm'][2],
            'border-radius': '12px',
            'padding': '4px 8px',
            'font-size': '12px',
            'font-weight': 'bold'
        },

        'badge_success': {
            'background-color': NORD_COLORS['aurora'][3],
            'color': NORD_COLORS['snow_storm'][2],
            'border-radius': '12px',
            'padding': '4px 8px',
            'font-size': '12px',
            'font-weight': 'bold'
        },

        'badge_warning': {
            'background-color': NORD_COLORS['aurora'][2],
            'color': NORD_COLORS['polar_night'][0],
            'border-radius': '12px',
            'padding': '4px 8px',
            'font-size': '12px',
            'font-weight': 'bold'
        },

        'badge_danger': {
            'background-color': NORD_COLORS['aurora'][0],
            'color': NORD_COLORS['snow_storm'][2],
            'border-radius': '12px',
            'padding': '4px 8px',
            'font-size': '12px',
            'font-weight': 'bold'
        }
    }

def get_plotly_theme():
    """
    Get a Plotly theme configuration using Nord colors.

    Returns:
        Dict: Plotly theme configuration
    """
    return {
        'layout': {
            'template': 'plotly_dark',
            'paper_bgcolor': NORD_COLORS['polar_night'][1],
            'plot_bgcolor': NORD_COLORS['polar_night'][1],
            'font': {
                'color': NORD_COLORS['snow_storm'][2],
                'family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
            },
            'colorway': [
                NORD_COLORS['frost'][2],    # Primary blue
                NORD_COLORS['aurora'][3],   # Green
                NORD_COLORS['aurora'][0],   # Red
                NORD_COLORS['aurora'][2],   # Yellow
                NORD_COLORS['aurora'][4],   # Purple
                NORD_COLORS['frost'][0],    # Cyan
                NORD_COLORS['aurora'][1],   # Orange
                NORD_COLORS['frost'][3]     # Dark blue
            ],
            'xaxis': {
                'gridcolor': NORD_COLORS['polar_night'][3],
                'linecolor': NORD_COLORS['polar_night'][3],
                'tickcolor': NORD_COLORS['snow_storm'][0],
                'color': NORD_COLORS['snow_storm'][1]
            },
            'yaxis': {
                'gridcolor': NORD_COLORS['polar_night'][3],
                'linecolor': NORD_COLORS['polar_night'][3],
                'tickcolor': NORD_COLORS['snow_storm'][0],
                'color': NORD_COLORS['snow_storm'][1]
            },
            'legend': {
                'bgcolor': NORD_COLORS['polar_night'][2],
                'bordercolor': NORD_COLORS['polar_night'][3],
                'borderwidth': 1,
                'font': {'color': NORD_COLORS['snow_storm'][2]}
            }
        }
    }

def get_custom_css():
    """
    Get custom CSS styles for enhanced visual appearance.

    Returns:
        str: CSS string with custom styles
    """
    return f"""
    /* Custom Nord-themed styles */
    
    /* Scrollbar styles */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {NORD_COLORS['polar_night'][1]};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {NORD_COLORS['polar_night'][3]};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {NORD_COLORS['frost'][3]};
    }}
    
    /* Input focus styles */
    .form-control:focus {{
        border-color: {NORD_COLORS['frost'][2]} !important;
        box-shadow: 0 0 0 0.2rem {NORD_COLORS['frost'][2]}40 !important;
        background-color: {NORD_COLORS['polar_night'][1]} !important;
        color: {NORD_COLORS['snow_storm'][2]} !important;
    }}
    
    /* Input and textarea base styling */
    .form-control {{
        background-color: {NORD_COLORS['polar_night'][1]} !important;
        border: 2px solid {NORD_COLORS['polar_night'][3]} !important;
        color: {NORD_COLORS['snow_storm'][2]} !important;
        border-radius: 6px !important;
    }}
    
    .form-control::placeholder {{
        color: {NORD_COLORS['snow_storm'][0]} !important;
        opacity: 0.8;
    }}
    
    /* Button hover effects */
    .btn:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    
    /* Card hover effects */
    .card:hover {{
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        transition: box-shadow 0.3s ease;
    }}
    
    /* Dropdown menu styles - Fixed for better visibility */
    .Select-control {{
        background-color: {NORD_COLORS['polar_night'][1]} !important;
        border: 2px solid {NORD_COLORS['polar_night'][3]} !important;
        border-radius: 6px !important;
        color: {NORD_COLORS['snow_storm'][2]} !important;
    }}
    
    .Select-placeholder {{
        color: {NORD_COLORS['snow_storm'][0]} !important;
    }}
    
    .Select-value-label {{
        color: {NORD_COLORS['snow_storm'][2]} !important;
    }}
    
    .Select-arrow-zone {{
        color: {NORD_COLORS['snow_storm'][1]} !important;
    }}
    
    .Select-menu-outer {{
        background-color: {NORD_COLORS['polar_night'][1]} !important;
        border: 2px solid {NORD_COLORS['polar_night'][3]} !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }}
    
    .Select-option {{
        background-color: {NORD_COLORS['polar_night'][1]} !important;
        color: {NORD_COLORS['snow_storm'][2]} !important;
        padding: 8px 12px !important;
    }}
    
    .Select-option:hover {{
        background-color: {NORD_COLORS['frost'][2]}40 !important;
        color: {NORD_COLORS['snow_storm'][2]} !important;
    }}
    
    .Select-option.is-focused {{
        background-color: {NORD_COLORS['frost'][2]}40 !important;
        color: {NORD_COLORS['snow_storm'][2]} !important;
    }}
    
    .Select-option.is-selected {{
        background-color: {NORD_COLORS['frost'][2]} !important;
        color: {NORD_COLORS['snow_storm'][2]} !important;
    }}
    
    /* Dash dropdown fix */
    .dash-dropdown .Select-value {{
        color: {NORD_COLORS['snow_storm'][2]} !important;
    }}
    
    .dash-dropdown .Select-single-value {{
        color: {NORD_COLORS['snow_storm'][2]} !important;
    }}
    
    /* Animation classes */
    .fade-in {{
        animation: fadeIn 0.5s ease-in;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .slide-in {{
        animation: slideIn 0.3s ease-out;
    }}
    
    @keyframes slideIn {{
        from {{ transform: translateX(-20px); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}
    
    /* Code syntax highlighting hint colors */
    .code-keyword {{ color: {NORD_COLORS['frost'][2]}; }}
    .code-string {{ color: {NORD_COLORS['aurora'][3]}; }}
    .code-comment {{ color: {NORD_COLORS['snow_storm'][0]}; font-style: italic; }}
    .code-number {{ color: {NORD_COLORS['aurora'][4]}; }}
    .code-function {{ color: {NORD_COLORS['frost'][1]}; }}
    
    /* Loading spinner */
    .loading-spinner {{
        border: 3px solid {NORD_COLORS['polar_night'][3]};
        border-top: 3px solid {NORD_COLORS['frost'][2]};
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }}
    
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* Stream scrubber styles */
    .stream-scrubber {{
        position: relative;
        height: 40px;
        background: {NORD_COLORS['polar_night'][2]};
        border-radius: 8px;
        cursor: pointer;
        margin: 10px 0;
    }}
    
    .stream-scrubber-track {{
        position: absolute;
        top: 50%;
        left: 10px;
        right: 10px;
        height: 4px;
        background: {NORD_COLORS['polar_night'][3]};
        border-radius: 2px;
        transform: translateY(-50%);
    }}
    
    .stream-scrubber-progress {{
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        background: {NORD_COLORS['frost'][2]};
        border-radius: 2px;
        transition: width 0.1s ease;
    }}
    
    .stream-scrubber-handle {{
        position: absolute;
        top: 50%;
        width: 20px;
        height: 20px;
        background: {NORD_COLORS['frost'][2]};
        border: 2px solid {NORD_COLORS['snow_storm'][2]};
        border-radius: 50%;
        transform: translate(-50%, -50%);
        cursor: grab;
        transition: all 0.2s ease;
    }}
    
    .stream-scrubber-handle:hover {{
        background: {NORD_COLORS['frost'][1]};
        transform: translate(-50%, -50%) scale(1.2);
    }}
    
    .stream-scrubber-handle:active {{
        cursor: grabbing;
        transform: translate(-50%, -50%) scale(1.1);
    }}
    
    .warmup-marker {{
        position: absolute;
        top: 0;
        width: 2px;
        height: 100%;
        background: {NORD_COLORS['aurora'][2]};
        border-radius: 1px;
        z-index: 2;
    }}
    
    .warmup-marker::after {{
        content: 'Warm-up';
        position: absolute;
        top: -25px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 10px;
        color: {NORD_COLORS['aurora'][2]};
        font-weight: bold;
        white-space: nowrap;
    }}
    
    /* Timeline info display */
    .timeline-info {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 12px;
        color: {NORD_COLORS['snow_storm'][1]};
        margin-top: 5px;
    }}
    
    .phase-indicator {{
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
    }}
    
    .phase-learning {{
        background: {NORD_COLORS['aurora'][2]}40;
        color: {NORD_COLORS['aurora'][2]};
    }}
    
    .phase-conformance {{
        background: {NORD_COLORS['frost'][2]}40;
        color: {NORD_COLORS['frost'][2]};
    }}
    """

# Utility functions for dynamic styling
def get_status_color(status: str) -> str:
    """Get color for status indicators."""
    status_colors = {
        'success': NORD_COLORS['aurora'][3],
        'warning': NORD_COLORS['aurora'][2],
        'error': NORD_COLORS['aurora'][0],
        'info': NORD_COLORS['frost'][2],
        'pending': NORD_COLORS['frost'][0],
        'inactive': NORD_COLORS['snow_storm'][0]
    }
    return status_colors.get(status.lower(), NORD_COLORS['snow_storm'][1])

def get_gradient_style(color1: str, color2: str, direction: str = 'to right') -> str:
    """Create a gradient background style."""
    return f"linear-gradient({direction}, {color1}, {color2})"

def get_nord_color_by_name(color_name: str, shade: int = 0) -> str:
    """
    Get a Nord color by name and shade.

    Args:
        color_name (str): Color family name ('polar_night', 'snow_storm', 'frost', 'aurora')
        shade (int): Shade index (0-3 for most families)

    Returns:
        str: Hex color code
    """
    if color_name in NORD_COLORS and 0 <= shade < len(NORD_COLORS[color_name]):
        return NORD_COLORS[color_name][shade]
    return NORD_COLORS['snow_storm'][1]  # Default fallback