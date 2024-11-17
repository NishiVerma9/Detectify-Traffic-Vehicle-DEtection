FROM gcr.io/google-appengine/python


RUN virtualenv /env -p python3

# Setting these environment variables are the same as running
# source /env/bin/activate.
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

RUN apt-get update && apt-get install -y python-opencv


ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Add the application source code.
ADD . /app


CMD gunicorn -b :$PORT main:app
