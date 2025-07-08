# Docker Setup for Zenodo Image Browser

This document explains how to run the Zenodo Image Browser using Docker Compose for a unified development environment.

## 🐳 Quick Start

### Prerequisites
- Docker Desktop installed and running
- At least 8GB RAM available for Docker (Ollama models can be large)

### One-Command Startup
```bash
# Start everything in development mode
./scripts/dev.sh start
```

Your application will be available at:
- **Frontend**: http://localhost:8501
- **Ollama API**: http://localhost:11434

## 📁 Docker Files Overview

- `docker-compose.yml` - Production configuration
- `docker-compose.dev.yml` - Development configuration with hot reloading
- `Dockerfile.frontend` - Streamlit frontend container
- `.dockerignore` - Optimizes build by excluding unnecessary files
- `scripts/dev.sh` - Development helper script

## 🚀 Development Workflow

### Starting the Application
```bash
# Start all services
./scripts/dev.sh start

# Or use docker-compose directly
docker-compose -f docker-compose.dev.yml up --build
```

### Stopping Services
```bash
# Stop all services
./scripts/dev.sh stop

# Or use docker-compose directly
docker-compose -f docker-compose.dev.yml down
```

### Viewing Logs
```bash
# Follow logs from all services
./scripts/dev.sh logs

# Or view specific service logs
docker-compose -f docker-compose.dev.yml logs -f frontend
docker-compose -f docker-compose.dev.yml logs -f ollama
```

### Installing Ollama Models
```bash
# Install a model (e.g., llama2:7b)
./scripts/dev.sh install-model llama2:7b

# Or install directly
docker-compose -f docker-compose.dev.yml exec ollama ollama pull llama2:7b
```

### Accessing Container Shell
```bash
# Open shell in frontend container
./scripts/dev.sh shell
```

## 🔧 Configuration

### Environment Variables
The following environment variables can be customized:

**Frontend (Streamlit)**:
- `STREAMLIT_SERVER_PORT` - Port for Streamlit (default: 8501)
- `STREAMLIT_SERVER_ADDRESS` - Bind address (default: 0.0.0.0)
- `STREAMLIT_SERVER_RUN_ON_SAVE` - Auto-reload on file changes (default: true)

**Ollama**:
- `OLLAMA_HOST` - Bind address for Ollama API (default: 0.0.0.0)

### Volumes
- `./` → `/app` - Application code (with hot reloading)
- `ollama_data` → `/root/.ollama` - Ollama models and data

### Networks
- `zenodo-network` - Internal network for service communication

## 🛠️ Troubleshooting

### Common Issues

**Port Already in Use**:
```bash
# Check what's using the port
lsof -i :8501
lsof -i :11434

# Stop conflicting services
docker-compose -f docker-compose.dev.yml down
```

**Out of Memory**:
```bash
# Check Docker memory usage
docker stats

# Increase Docker memory limit in Docker Desktop settings
# Recommended: 8GB+ for Ollama models
```

**Ollama Model Not Found**:
```bash
# List installed models
docker-compose -f docker-compose.dev.yml exec ollama ollama list

# Install missing model
./scripts/dev.sh install-model llama2:7b
```

**Build Cache Issues**:
```bash
# Clean build cache
docker-compose -f docker-compose.dev.yml build --no-cache

# Or clean everything
./scripts/dev.sh clean
```

### Performance Tips

1. **Use .dockerignore**: Excludes unnecessary files from build context
2. **Layer Caching**: Requirements are installed before copying code
3. **Volume Mounting**: Code changes trigger hot reload without rebuild
4. **Resource Limits**: Monitor Docker Desktop memory allocation

## 🔄 Migration from Local Development

If you're currently running the app locally:

1. **Stop local services**:
   ```bash
   # Stop Streamlit
   Ctrl+C
   
   # Stop Ollama
   brew services stop ollama
   ```

2. **Start with Docker**:
   ```bash
   ./scripts/dev.sh start
   ```

3. **Verify everything works**:
   - Frontend: http://localhost:8501
   - Ollama models: Check the AI Assistant section

## 🚀 Production Deployment

For production deployment, use the main `docker-compose.yml`:

```bash
# Production build
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 📊 Monitoring

### Health Checks
- Frontend: http://localhost:8501/_stcore/health
- Ollama: http://localhost:11434/api/tags

### Resource Usage
```bash
# Monitor container resources
docker stats

# View container details
docker-compose -f docker-compose.dev.yml ps
```

## 🔮 Future Enhancements

The Docker setup is designed to easily accommodate:

- **FastAPI Backend**: Uncomment the backend service in docker-compose.yml
- **Database**: Add PostgreSQL or Redis services
- **Monitoring**: Add Prometheus/Grafana containers
- **CI/CD**: Use the same Dockerfiles in GitHub Actions

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit Docker Guide](https://docs.streamlit.io/knowledge-base/deploy/docker)
- [Ollama Docker Guide](https://ollama.ai/docs/guides/docker) 