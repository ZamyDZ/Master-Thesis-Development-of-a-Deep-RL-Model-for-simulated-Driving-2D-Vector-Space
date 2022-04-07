FROM ubuntu:20.04

ENV TZ=Europe/Berlin
# this fixes the problem with the geographic location input
ENV DEBIAN_FRONTEND noninteractive

# apt-get upgrade -y && \
RUN mkdir masterthesis && \
    mkdir masterthesis/DonkeySimLinux && \
    mkdir masterthesis/src && \
    apt-get update && \
    apt-get install nano && \
    apt install sublime-text && \
    apt install lubuntu-desktop lightdm && \
    apt-get install build-essential -y && \ 
    apt-get install python3 -y && \
    apt install python3-pip -y && \
    apt install curl -y && \
    apt install git-all -y && \
    pip3 install notebook && \
    pip3 install opencv-python && \
    pip3 install numpy && \
    pip3 install pandas && \
    pip3 install -U matplotlib && \
    pip3 install seaborn && \
    pip3 install glob2 && \
    pip3 install gym

#WORKDIR /masterthesis

# ADD is for folders and COPY for single files
#COPY DonkeySimLinux /masterthesis/DonkeySimLinux/
COPY /src/birds_eye_vector_space.py /masterthesis/src/
COPY /src/startSim.sh /masterthesis/src/