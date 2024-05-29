#Base Image to use
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

#install all requirements in requirements.txt
RUN pip install --upgrade pip 
RUN pip install -r app/requirements.txt 

#Expose port 8080
EXPOSE 8080

# healthcheck the exposed port
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

#Copy all files in current directory into app directory
COPY . /app

# #Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "./app/🏠_Home.py", "--server.port=8080", "--server.address=0.0.0.0"]