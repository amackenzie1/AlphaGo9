# Pull the base image with python 3.8 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.8

# Install OS packages for Pillow-SIMD
RUN yum install vi

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./
COPY *.py ./
COPY baby_alphazero ./


# Install the python requirements from requirements.txt
RUN python3.8 -m pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]

