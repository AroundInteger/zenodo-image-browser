import streamlit.web.cli as stcli
import sys
import os
 
if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=localhost"]
    sys.exit(stcli.main()) 