FROM continuumio/miniconda3

WORKDIR /app

ADD . /app

RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "pygenv", "/bin/bash", "-c"]

ENTRYPOINT ["sh"]
