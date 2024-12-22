FROM python@sha256:320a7a4250aba4249f458872adecf92eea88dc6abd2d76dc5c0f01cac9b53990

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade \
"pip>=23.3" \
"setuptools>=70.0"

# Copy our application code
WORKDIR /var/app

COPY . .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

EXPOSE 80

# Start the app
CMD ["gunicorn", "-b", "0.0.0.0:80","challenge.api:app","--workers","1","--timeout","600","-k","uvicorn.workers.UvicornWorker"]