#!/bin/bash

# Development script for Zenodo Image Browser
# Usage: ./scripts/dev.sh [command]

set -e

case "$1" in
    "start")
        echo "🚀 Starting Zenodo Image Browser in development mode..."
        docker-compose -f docker-compose.dev.yml up --build
        ;;
    "stop")
        echo "🛑 Stopping services..."
        docker-compose -f docker-compose.dev.yml down
        ;;
    "restart")
        echo "🔄 Restarting services..."
        docker-compose -f docker-compose.dev.yml down
        docker-compose -f docker-compose.dev.yml up --build
        ;;
    "logs")
        echo "📋 Showing logs..."
        docker-compose -f docker-compose.dev.yml logs -f
        ;;
    "clean")
        echo "🧹 Cleaning up Docker resources..."
        docker-compose -f docker-compose.dev.yml down -v
        docker system prune -f
        ;;
    "shell")
        echo "🐚 Opening shell in frontend container..."
        docker-compose -f docker-compose.dev.yml exec frontend /bin/bash
        ;;
    "install-model")
        if [ -z "$2" ]; then
            echo "❌ Please specify a model name (e.g., llama2:7b)"
            exit 1
        fi
        echo "📦 Installing Ollama model: $2"
        docker-compose -f docker-compose.dev.yml exec ollama ollama pull "$2"
        ;;
    *)
        echo "Zenodo Image Browser - Development Script"
        echo ""
        echo "Usage: ./scripts/dev.sh [command]"
        echo ""
        echo "Commands:"
        echo "  start         - Start the application in development mode"
        echo "  stop          - Stop all services"
        echo "  restart       - Restart all services"
        echo "  logs          - Show logs from all services"
        echo "  clean         - Clean up Docker resources"
        echo "  shell         - Open shell in frontend container"
        echo "  install-model <model> - Install an Ollama model"
        echo ""
        echo "Examples:"
        echo "  ./scripts/dev.sh start"
        echo "  ./scripts/dev.sh install-model llama2:7b"
        ;;
esac 