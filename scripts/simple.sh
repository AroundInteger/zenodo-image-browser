#!/bin/bash

# Simple development script for Zenodo Image Browser
# Runs Ollama in Docker, Streamlit locally
# Usage: ./scripts/simple.sh [command]

set -e

case "$1" in
    "start")
        echo "🚀 Starting Ollama in Docker..."
        docker-compose -f docker-compose.simple.yml up -d
        
        echo "⏳ Waiting for Ollama to start..."
        sleep 5
        
        echo "🤖 Starting Streamlit locally..."
        echo "Your app will be available at: http://localhost:8501"
        echo "Ollama API will be available at: http://localhost:11434"
        echo ""
        echo "Press Ctrl+C to stop both services"
        
        # Start Streamlit locally
        streamlit run app.py
        ;;
    "stop")
        echo "🛑 Stopping Ollama..."
        docker-compose -f docker-compose.simple.yml down
        ;;
    "restart")
        echo "🔄 Restarting Ollama..."
        docker-compose -f docker-compose.simple.yml restart
        ;;
    "logs")
        echo "📋 Showing Ollama logs..."
        docker-compose -f docker-compose.simple.yml logs -f
        ;;
    "install-model")
        if [ -z "$2" ]; then
            echo "❌ Please specify a model name (e.g., llama2:7b)"
            exit 1
        fi
        echo "📦 Installing Ollama model: $2"
        docker-compose -f docker-compose.simple.yml exec ollama ollama pull "$2"
        ;;
    "status")
        echo "📊 Service Status:"
        docker-compose -f docker-compose.simple.yml ps
        echo ""
        echo "Installed models:"
        docker-compose -f docker-compose.simple.yml exec ollama ollama list 2>/dev/null || echo "Ollama not running"
        ;;
    *)
        echo "Zenodo Image Browser - Simple Development Script"
        echo ""
        echo "This script runs Ollama in Docker and Streamlit locally."
        echo "Perfect for development when you want to use your existing Python environment."
        echo ""
        echo "Usage: ./scripts/simple.sh [command]"
        echo ""
        echo "Commands:"
        echo "  start         - Start Ollama in Docker, then Streamlit locally"
        echo "  stop          - Stop Ollama Docker container"
        echo "  restart       - Restart Ollama Docker container"
        echo "  logs          - Show Ollama logs"
        echo "  install-model <model> - Install an Ollama model"
        echo "  status        - Show service status and installed models"
        echo ""
        echo "Examples:"
        echo "  ./scripts/simple.sh start"
        echo "  ./scripts/simple.sh install-model llama2:7b"
        echo "  ./scripts/simple.sh status"
        ;;
esac 