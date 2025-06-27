# Streaming Process Mining Challenge Platform

## 🚀 Enhanced Platform Features

The platform has been upgraded from a simple testing interface to a comprehensive challenge management system:

### ✨ New Features

1. **Professional File Upload System**
   - Upload Python files (.py) or ZIP archives (.zip)
   - Automatic algorithm detection and validation
   - Library dependency specification
   - Session-based file management

2. **Comprehensive Leaderboard**
   - Real-time ranking based on multiple metrics
   - Performance distribution visualization
   - Team submission history
   - Detailed evaluation results

3. **Advanced Submission Management**
   - Background evaluation processing
   - Persistent submission database
   - Status tracking and notifications
   - Comprehensive error handling

4. **Enhanced User Interface**
   - Tabbed navigation (Test & Submit, Leaderboard, My Submissions, Documentation)
   - Nord-themed professional styling
   - Interactive timeline scrubber
   - Fullscreen result visualization

## 📦 Installation & Setup

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install required packages
pip install -r requirements.txt
```

### Create `requirements.txt`

```txt
dash>=2.14.0
dash-bootstrap-components>=1.5.0
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0
sqlalchemy>=2.0.0
pm4py>=2.7.0  # Optional: for PNML file support
scikit-learn>=1.3.0  # Optional: for advanced algorithms
```

### Directory Structure

```
streaming-process-mining-challenge/
├── app.py                          # Main enhanced application
├── requirements.txt                # Python dependencies
├── assets/
│   ├── config/
│   │   └── stream_configuration.json
│   └── data/
│       ├── model_one.pnml         # Optional PNML files
│       └── model_two.pnml
├── src/
│   ├── functions.py               # Utility functions
│   └── ui/
│       ├── algorithm_base.py      # Base algorithm class
│       ├── styles.py             # Nord theme styles
│       ├── stream_manager.py     # Stream generation
│       └── test_streams.py       # Concept drift streams
├── leaderboard_system.py         # Submission & evaluation management
├── leaderboard_ui.py             # Leaderboard UI components
├── file_upload_utils.py          # File upload utilities
├── example_algorithm.py          # Example algorithm implementation
├── uploaded_algorithms/           # Directory for uploaded files
├── challenge_submissions.db       # SQLite database (auto-created)
└── README.md                     # This file
```

### Quick Start

1. **Clone and Setup**
```bash
git clone <repository-url>
cd streaming-process-mining-challenge
pip install -r requirements.txt
```

2. **Create Configuration Files**

Create `assets/config/stream_configuration.json`:
```json
{
  "streams": [
    {
      "id": "simple_sequential",
      "name": "Simple Concept Drift",
      "description": "A basic stream containing a simple concept drift. The first 60% of the stream follows one model, and the last 40% follows another.",
      "type": "pnml_drift",
      "is_testing": true,
      "pnml_log_a": "data/model_one.pnml",
      "pnml_log_b": "data/model_two.pnml",
      "drift_config": {
        "drift_type": "sudden",
        "drift_begin_percentage": 0.6,
        "drift_end_percentage": 0.6,
        "event_level_drift": false,
        "num_cases_a": 300,
        "num_cases_b": 200
      }
    }
  ]
}
```

3. **Run the Application**
```bash
python app.py
```

4. **Access the Platform**
   - Open browser to `http://localhost:8050`
   - Navigate through the tabs to explore features

## 🧪 Creating and Testing Algorithms

### Algorithm Implementation

Create a Python file that extends `BaseAlgorithm`:

```python
from algorithm_base import BaseAlgorithm
from typing import Dict, Any

class MyConformanceAlgorithm(BaseAlgorithm):
    def __init__(self):
        super().__init__()
        # Initialize your algorithm here
        self.patterns = {}
        
    def learn(self, event: Dict[str, Any]) -> None:
        """Learn from event during warm-up phase."""
        self.learning_events += 1
        
        case_id = event.get('case:concept:name')
        activity = event.get('concept:name')
        
        # Your learning logic here
        if case_id and activity:
            # Learn patterns from the event
            pass
    
    def conformance(self, event: Dict[str, Any]) -> float:
        """Return conformance score (0.0 to 1.0)."""
        self.conformance_events += 1
        
        case_id = event.get('case:concept:name')
        activity = event.get('concept:name')
        
        # Your conformance checking logic here
        if case_id and activity:
            # Calculate and return conformance score
            return 1.0  # Replace with actual calculation
        
        return 0.5  # Default neutral score
```

### Event Structure

Events are dictionaries with these standard fields:

```python
event = {
    'case:concept:name': 'case_001',    # Case identifier
    'concept:name': 'Create Application', # Activity name
    'time:timestamp': '2023-01-01 10:00:00', # Event timestamp
    'concept:origin': 'Log A'           # Origin log (for drift streams)
    # Additional domain-specific attributes may be present
}
```

### Testing Your Algorithm

1. **Upload Algorithm**
   - Go to "Test & Submit" tab
   - Click "Upload Algorithm"
   - Select your Python file or ZIP archive
   - Specify any required libraries

2. **Configure Test**
   - Select a test stream
   - Choose number of cases (50-1000)
   - Click "Run Test"

3. **View Results**
   - Interactive conformance plot
   - Performance metrics
   - Comparison with baseline
   - Fullscreen analysis option

### Submitting to Challenge

1. **Test First**: Always test your algorithm before submitting
2. **Fill Submission Form**: 
   - Team name
   - Contact email
   - Algorithm name
   - Description
3. **Submit**: Click "Submit to Challenge"
4. **Monitor Progress**: Check "My Submissions" tab for evaluation status

## 📊 Understanding Evaluation Metrics

### Core Metrics

1. **Accuracy**: Percentage of predictions within 0.1 of baseline
2. **MAE (Mean Absolute Error)**: Average absolute difference from baseline
3. **RMSE (Root Mean Square Error)**: Square root of mean squared differences
4. **Speed Score**: Normalized processing time per event
5. **Robustness Score**: Based on number of major errors (>0.3 difference)

### Composite Score Calculation

```python
composite_score = (
    0.30 * accuracy_score +      # 30% weight
    0.25 * mae_score +           # 25% weight  
    0.20 * rmse_score +          # 20% weight
    0.15 * speed_score +         # 15% weight
    0.10 * robustness_score      # 10% weight
)
```

### Leaderboard Ranking

- **Rank 1-3**: Medal display (🥇🥈🥉)
- **Rank 4-10**: Highlighted in leaderboard
- **All submissions**: Tracked with detailed metrics

## 🛠️ Advanced Configuration

### Adding New Test Streams

Edit `assets/config/stream_configuration.json`:

```json
{
  "streams": [
    {
      "id": "your_new_stream",
      "name": "Custom Stream Name",
      "description": "Description of your stream...",
      "type": "pnml_drift",
      "is_testing": true,
      "pnml_log_a": "data/your_model_a.pnml",
      "pnml_log_b": "data/your_model_b.pnml",
      "drift_config": {
        "drift_type": "gradual",        # "sudden" or "gradual"
        "drift_begin_percentage": 0.3,  # When drift starts (0.0-1.0)
        "drift_end_percentage": 0.7,    # When drift ends (0.0-1.0)
        "event_level_drift": true,      # Event-level vs case-level
        "num_cases_a": 250,
        "num_cases_b": 250
      }
    }
  ]
}
```

### Database Configuration

The platform uses SQLite by default. For production deployments:

```python
# In leaderboard_system.py, modify SubmissionDatabase initialization
submission_manager = SubmissionManager(
    upload_directory="uploaded_algorithms",
    database_path="postgresql://user:pass@localhost/challenge_db"  # For PostgreSQL
)
```

### Customizing Evaluation

Modify evaluation parameters in `leaderboard_system.py`:

```python
class EvaluationEngine:
    def __init__(self, stream_manager, upload_manager, algorithm_loader):
        # ...
        self.test_streams = ['stream1', 'stream2', 'stream3']
        self.test_cases = [100, 250, 500, 1000]
        self.timeout_seconds = 600  # 10 minutes
```

## 🔧 Troubleshooting

### Common Issues

1. **Algorithm Not Detected**
   - Ensure your class extends `BaseAlgorithm`
   - Check that `learn()` and `conformance()` methods are implemented
   - Verify file encoding is UTF-8

2. **Import Errors**
   - List all required libraries in the upload dialog
   - Use standard library names (e.g., 'numpy', not 'numpy==1.24.0')
   - Avoid relative imports in uploaded files

3. **Evaluation Failures**
   - Check algorithm doesn't use prohibited operations
   - Ensure methods return correct types (float for conformance)
   - Add error handling for edge cases

4. **Performance Issues**
   - Optimize algorithm for large event streams
   - Avoid expensive operations in conformance checking
   - Consider memory usage for long-running evaluations

### Debug Mode

Run with debug enabled:

```bash
python app.py
# or
DASH_DEBUG=True python app.py
```

### Logging

Add logging to your algorithm:

```python
import logging

class MyAlgorithm(BaseAlgorithm):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def learn(self, event):
        self.logger.debug(f"Learning from event: {event}")
        # Your implementation
```

## 🚀 Production Deployment

### Environment Variables

```bash
export DASH_ENV=production
export DASH_DEBUG=False
export DATABASE_URL=postgresql://...
export UPLOAD_DIRECTORY=/var/uploads
export SECRET_KEY=your-secret-key
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8050

CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app:server"]
```

### Security Considerations

1. **File Upload Security**
   - Validate file types and sizes
   - Scan uploaded code for malicious patterns
   - Run evaluations in sandboxed environments

2. **Database Security**
   - Use environment variables for database credentials
   - Implement user authentication if needed
   - Regular database backups

3. **Rate Limiting**
   - Limit submission frequency per team
   - Implement evaluation queue management
   - Monitor system resources

## 📈 Monitoring & Analytics

### Submission Analytics

```python
# Get platform statistics
stats = submission_manager.leaderboard_manager.get_leaderboard_stats()
print(f"Total submissions: {stats['total_submissions']}")
print(f"Completion rate: {stats['completion_rate']:.1%}")
```

### Performance Monitoring

```python
# Monitor evaluation performance
evaluations = submission_manager.database.get_evaluations(submission_id)
for eval_result in evaluations:
    print(f"Stream: {eval_result.stream_id}, Runtime: {eval_result.total_runtime}s")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

[Your License Here]

## 📞 Support

For questions and support:
- Create an issue in the repository
- Contact the challenge organizers
- Check the documentation tab in the platform

---

🎉 **Ready to start the challenge!** Upload your algorithm and see how it performs against others in real-time streaming process mining scenarios.