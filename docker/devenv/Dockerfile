# Available images: 
# $ gcloud container images list --repository="gcr.io/deeplearning-platform-release"
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-12

ENV SHELL=/bin/bash

WORKDIR /usr/src

# Container doesn't have UNZIP or less
RUN apt-get update
RUN apt-get install unzip
RUN apt-get install less

# Installing requirements, including shipped submodules
COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install --upgrade pip==21.3.1
RUN pip install --no-cache-dir -r requirements.txt

# Installing BODA as a package
ADD . /home/ubuntu/boda2/
WORKDIR /home/ubuntu/boda2/
RUN pip install -e .

# Install MEME suite
RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    openmpi-bin \
    ghostscript \
    libgs-dev \
    libgd-dev \
    libexpat1-dev \
    zlib1g-dev \
    libxml2-dev \
    autoconf automake libtool \
    libhtml-template-compiled-perl \
    libxml-opml-simplegen-perl \
    libxml-libxml-debugging-perl
    
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install Log::Log4perl'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install Math::CDF'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install CGI'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install HTML::PullParser'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install HTML::Template'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::Simple'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::Parser::Expat'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::LibXML'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::LibXML::Simple'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::Compile'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::Compile::SOAP11'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::Compile::WSDL11'
RUN PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install XML::Compile::Transport::SOAPHTTP'

WORKDIR /opt/meme/
RUN wget https://meme-suite.org/meme/meme-software/5.4.0/meme-5.4.0.tar.gz
RUN tar zxf meme-5.4.0.tar.gz
WORKDIR /opt/meme/meme-5.4.0
RUN ./configure --prefix=/opt/meme --enable-build-libxml2 --enable-build-libxslt
RUN make && make install
    
ENV PATH=/opt/meme/bin:/opt/meme/libexec/meme-5.4.0:$PATH

# Go to root
WORKDIR /
