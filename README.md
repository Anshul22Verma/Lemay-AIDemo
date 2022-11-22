# Lemay-AIDemo
A demo for lemay AI assignment tasks. 


# 2. Docker for HuggingFace model inference
## a)
Work inside the directory `inference` 
### Base Docker from
https://hub.docker.com/r/nvidia/cuda/tags, matches the main installations in the used AWS AMI https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-1-11-ubuntu-20-04/.

### Run
To create the docker image run the command
```docker run -t squad .```
To host the docker in the instance run the command, port 80 is opened in the security and is used in the docker file as well to run this app
```docker run -p 80:80 --cpus 2 squad```

### Also moved the docker to ECR and used a ECS Fargate to host this (to save cost)
public ip `3.235.222.30`

## b)
The docker is running the app using gunicorn.

## c)
## Test 
### Payload for the POST request
dummy payload in `payload.json`

### CRUL to make the POST request with the JSON
```curl -X POST http://<publicIP>/answer -H "Content-Type: application/json" -d @payload.json```

### Sample Notebook
`inference.ipynb`

## d)
Wanted to try the pipeline for a contex based Q&A model.


# 3. 
## a)
Exploratory analysis's work is in dir `analysis`. It uses `squad` dataset which was used to train the model used to create the server for inference.

## b)
Notebook `analysis/exploratory_analysis.ipynb`

## c) 
Used this dataset because was using the model which was trained the model which peeked my curiosity.

## d)
Inference notebook was ran on Google Colab and can be rerun in colab easily.

Whereas the server was hosted in the DeepLearningAMI in AWS and uses the `requirements.txt` file for python dependencies.
