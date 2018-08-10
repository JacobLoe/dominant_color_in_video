FROM ubuntu:18.04

MAINTAINER Christian Hentschel "christian.hentschel@hpi.de"

RUN apt-get update && apt-get install -y \
    software-properties-common \ 
    #python-software-properties \
    python-opencv \
    python-numpy \
    python-pandas \
    python-scipy \
    lsb-release && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists


COPY extract_dominant_color.py /root/extract_dominant_color.py

VOLUME ["/data"]
WORKDIR /root
CMD [ "/root/extract_dominant_color.py", "--help" ]
