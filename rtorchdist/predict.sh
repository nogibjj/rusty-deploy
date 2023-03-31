#!/bin/bash

curl -X POST \
     -H "Content-Type: multipart/form-data" \
     -F "image=@./lion.jpg" \
     http://127.0.0.1:8080/predict
