FROM python:3.9-slim

# Create app directory
WORKDIR /app
 
# Install app dependencies
COPY reqs.txt ./

RUN pip install --no-cache-dir --upgrade -r reqs.txt
 
# Bundle app source
COPY . .

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 