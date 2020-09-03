#preparing Ubuntu setup
FROM Ubuntu:18.04
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

#installing conda and preparing environment
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH "/miniconda/bin:$PATH"
RUN conda create -y -n Zesty jupyterlab numpy pandas -c anaconda -c conda-forge
RUN echo "source activate Zesty" > ~/.bashrc
ENV PATH "/miniconda/envs/Zesty/bin:$PATH"
