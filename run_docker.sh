#!/bin/sh

# Build docker image
docker build -t naivebayes-python .

# Run the image
docker run --rm naivebayes-python