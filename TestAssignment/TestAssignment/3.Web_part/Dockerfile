FROM ubuntu:16.04

MAINTAINER <<Ognjen Tanovic>>
RUN apt-get update -y && \
   apt-get install -y python3-pip python3-dev
COPY ./requirements.txt /3.Web_part/requirements.txt
WORKDIR /3.Web_part
RUN pip3 install -r requirements.txt
COPY . /3.Web_part
ENTRYPOINT [ "python3" ]
CMD [ "test_assignment_app.py" ]
