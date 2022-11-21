# Lemay-AIDemo
A demo for lemay AI assignment tasks. 


# 2. Docker for HuggingFace model inference
Work inside the directory `inference` 
### Base Docker from
https://hub.docker.com/r/nvidia/cuda/tags, matches the main installations in the used AWS AMI https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-1-11-ubuntu-20-04/.

### Run
To create the docker image run the command
```docker run -t squad .```
To host the docker in the instance run the command, port 80 is opened in the security and is used in the docker file as well to run this app
```docker run -p 80:80 --cpus 2 squad```

### Moving the docker to ECR and using a Fargate ECS to host this (to save cost)

## Test 
### Payload for the POST request
dummy payload in `payload.json`

### CRUL to make the POST request with the JSON
```curl -X POST http://54.90.166.57:80/answer -H "Content-Type: application/json" -d @payload.json```

### Sample Notebook
`inference.ipynb`