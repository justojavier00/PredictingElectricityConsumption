#preparing Ubuntu setup
FROM ubuntu:18.04
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl ca-certificates && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

#installing conda and preparing environment
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH "/miniconda/bin:$PATH"
RUN conda create -y -n Zesty jupyterlab numpy pandas scikit-learn -c anaconda -c conda-forge
RUN echo "source activate Zesty" > ~/.bashrc
ENV PATH "/miniconda/envs/Zesty/bin:$PATH"
RUN conda clean -y --all # clean conda to reduce size
RUN apt-get remove -y curl && apt-get autoremove -y #removing tools we do not need anylonger to reduce size

WORKDIR /root/
COPY --chown=root src/ /root/
