# selecting a base docker container with cuda and cudnn installed that maches the VM used to build the docker
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# setting up the environment 

# not sure if the base docker is regularly updated so adding the essential 
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    unzip \
    python3 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# I prefer having the model and code downloaded in the docker if the model and the code version is not changing
# but to be able to use CD we should pull the code everytime and download the model
WORKDIR /git
RUN git clone https://github.com/Anshul22Verma/Lemay-AIDemo.git /git/code
WORKDIR /git/code/inference

# upgrade and install the requirements
RUN pip3 install --no-cache-dir -Ur requirements.txt

# Envioronment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Expose a port to recieve requests
EXPOSE 5000

# Start the app, the prediction is being made in GPU and its using an instance with one GPU only
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app", "--workers", "1", "-k", "uvicorn.workers.UvicornWorker"]

# To handle in multiple requests I will attach the  an ALB to the ECS