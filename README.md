# Streaming Process Mining Challenge Platform

Streaming process mining deals with the real-time analysis of streaming data. Event streams require algorithms capable of processing data incrementally. To systematically address these challenges,
we propose the Adaptive Validation of Online Challenge Algorithms for Dynamic Outcomes (AVOCADO 🥑), providing standardized evaluation criteria for streaming algorithms. The AVOCADO 🥑 evaluates algorithms on streaming-specific metrics like accuracy, MAE, RMSE, Processing Latency, and robustness. This initiative seeks to foster innovation and community-driven discussions to advance the field of streaming process
mining.

[//]: # (Add a image of the platform here)

[![Streaming Process Mining Challenge Platform](docs/evaluating_drifts.png)](docs/evaluating_drifts.png)

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
# Core Dash dependencies
dash==2.14.1
dash-bootstrap-components==1.5.0
plotly==5.17.0

numpy==2.3.1
pandas==2.3.0

pm4py==2.7.11

python-dateutil==2.8.2

jupyter==1.0.0
ipython==8.15.0

gunicorn==23.0.0
waitress==2.1.2

# Optional: For testing
pytest==7.4.2

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

COPY .. .

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