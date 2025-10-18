# ==========================================
#   Copyright (c) 2025 Ayush Singh  #
# ==========================================

# -------------------------------------------------------------------
#   Mnemonic:   Dockerfile
#   Abstract:   Hello-World Flask REST-API Docker Template
#
#   Date:       18 Oct 2025
#   Author:     Ayush Singh
# -------------------------------------------------------------------

FROM tiangolo/uwsgi-nginx-flask:python3.8

# add dummy app
ENV INSTALL_PATH /usr/src/helloworld
RUN mkdir -p $INSTALL_PATH

# install net-tools mysql-client
# using mariadb-client inplace of mysql-client
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    net-tools \
  && apt-get install -y --no-install-recommends mariadb-client \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR $INSTALL_PATH

# logging addition is included
# change the directory `app` as required
RUN mkdir -p /tmp/logs/app/

# setting up environment variables
# application specific ENVIRONMENT VARIABLES
# can be either set from this Dockerfile
# or individual file can be configured under
# `.env` and `./app/main/.env` directory.
# getting started, specific environment variables
# are defined in Dockerfile for understanding as
# mentioned below:
#
# ENVIRONMENT VARIABLES for `.env`
#
# ENV port
# ENV host
# ENV PROJECT_ENV_NAME
#
# ENVIRONMENT VARIABLES for `./app/main/.env`
#
# ENV database_host
# ENV database_port
# ENV username
# ENV password
# ENV DATABASE_URL
# ENV dev_db
#
# NOTE - dummy `.env` files are present with
# an extension `.env.bkp`


# setup flask environment
# install all requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# copy all files and folder to docker
COPY . .

# run the application in docker environment
# you can use the wsgi service to start the application
# or the default python can also be used
# run create_db to create all the tables using flask
CMD [ "python", "./create_db.py" ]
CMD [ "python", "./manage.py" ]
