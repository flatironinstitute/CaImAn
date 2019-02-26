FROM continuumio/anaconda3

RUN apt-get update && apt-get install -y \
    gcc g++ \
    libgtk2.0-0 \
    xvfb \
    qtbase5-dev \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/jenkins/.conda && chown -R 996 /home/jenkins
