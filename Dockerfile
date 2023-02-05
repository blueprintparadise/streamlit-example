FROM python:3.10-slim


# install as a package
RUN pip install streamlit

EXPOSE 8501

# cmd for running the API
CMD ["python", "-m", "streamlit", "run", "application.py"]