# financeGard

A web application built with Flask.

## Quick Start

1. Activate virtual environment:
   ```bash
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
3. **Install dependencies** (if not already installed)
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

Your application will be available at `http://localhost:5000`

## 📁 Project Structure

```
financeGard/
├── venv/                   # Virtual environment
├── financeGard/             # Main application code
│   ├── api/                # API module
│   ├── auth/               # Authentication module
│   ├── models/             # Models module
│   ├── static/             # Static files (CSS, JS, images)
│       ├── uploads/
│   ├── templates/          # HTML templates (if web app)
│   └── app.py              # Main application file
├── tests/                  # Test files
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (local)
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── app.py                # Application runner
└── README.md             # This file
```

## 🛠️ Development

### Framework: Flask
Lightweight WSGI web application framework

### Application Type: API Only

### Available Endpoints
- `GET /` - API welcome message
- `GET /health` - Health check endpoint

### Adding New Routes

#### Flask specific instructions:
'''

        # Add framework-specific route examples
        if framework == 'flask':
            readme_content += '''
```python
from flask import Flask

@app.route('/new-route')
def new_route():
    return 'Hello from new route!'
```
'''
        elif framework == 'fastapi':
            readme_content += '''
```python
from fastapi import FastAPI

@app.get("/new-route")
async def new_route():
    return {"message": "Hello from new route!"}
```
'''
        elif framework == 'django':
            readme_content += '''
1. Add to `app/urls.py`:
```python
path('new-route/', views.new_route, name='new_route'),
```

2. Add to `app/views.py`:
```python
def new_route(request):
    return JsonResponse({'message': 'Hello from new route!'})
```
'''
        elif framework == 'bottle':
            readme_content += '''
```python
@app.route('/new-route')
def new_route():
    return {'message': 'Hello from new route!'}
```
'''
        elif framework == 'pyramid':
            readme_content += '''
1. Add route in `main()`:
```python
config.add_route('new_route', '/new-route')
```

2. Add view function:
```python
@view_config(route_name='new_route', renderer='json')
def new_route_view(request):
    return {'message': 'Hello from new route!'}
```


## 🧪 Testing

Run tests with:
```bash
pytest tests/
```

## 📦 Deployment

### Environment Variables
Make sure to set these in production:
- `SECRET_KEY`: A secure secret key
- `DEBUG`: Set to `False` in production
- `PORT`: Port number for the application

### Docker (Optional)
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .`
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [Amen CLI](https://github.com/your-username/amen-cli)
- Powered by Flask

---

Happy coding! 🎉
