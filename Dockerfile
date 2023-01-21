# Pull the base image with python 3.8 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.8

# Install OS packages for Pillow-SIMD
RUN yum -y install vi

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./
COPY *.py ./
COPY *.sh ./
COPY baby_alphazero baby_alphazero


# Install the python requirements from requirements.txt
RUN python3.8 -m pip install -r requirements.txt
# RUN flask --app server run --host 0.0.0.0 -p 1234

CMD ["app.lambda_handler"]

