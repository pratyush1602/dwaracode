# StackResolve

A comprehensive FastAPI-based application that provides multiple services including KYC verification, object detection, and log analysis.

## Features

- **KYC System**
  - User verification workflow
  - Admin dashboard for KYC management
  - Session-based verification process

- **Object Detection**
  - Real-time object detection capabilities
  - Support for multiple AI models
  - Configurable detection parameters

- **Log Analysis**
  - Advanced log processing and analysis
  - Pattern recognition in log files
  - Customizable analysis parameters

- **Configuration Management**
  - Dynamic configuration through JSON files
  - Model management and configuration
  - Backup system for configurations

## Prerequisites

- Python 3.8+
- Node.js (for certain frontend features)
- Virtual environment recommended

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd StackResolve
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Copy `.env.example` to `.env` (if provided)
- Configure your environment variables

## Project Structure

```
├── api/                  # API endpoints and routers
├── services/            # Core service implementations
├── templates/           # Jinja2 HTML templates
├── static/             # Static files (if any)
├── data/               # Data storage directory
├── json/               # Configuration JSON files
├── models/             # AI model files
├── main.py            # Main application entry point
└── requirements.txt    # Python dependencies
```

## Running the Application

1. Start the server:
```bash
python main.py
```

2. Access the application:
- Main interface: `http://localhost:3000`
- KYC interface: `http://localhost:3000/kyc`
- KYC Admin: `http://localhost:3000/kyc/admin`

## API Endpoints

- `/api/analyze/*` - Log analysis endpoints
- `/api/config/*` - Configuration management
- `/api/*` - Object detection endpoints
- `/api/kyc/*` - KYC verification endpoints

## Docker Support

The application can be containerized using Docker:

```bash
docker build -t stackresolve .
docker run -p 3000:3000 stackresolve
```

## Configuration

- Configuration files are stored in `json/config.json`
- Automatic configuration backups are maintained
- Model configurations can be updated through the config API

## Security Notes

- Ensure proper environment variable configuration
- Secure your API keys and sensitive data
- Configure CORS settings as needed in production

## Development

- Uses FastAPI for high-performance async API
- Jinja2 templating for server-side rendering
- Supports hot-reloading during development
- Includes comprehensive error handling

## License

[Your License Type] - See LICENSE file for details 