FROM  anibali/pytorch:no-cuda
MAINTAINER  Alvaro Gonzalez Jimenez <alvarogonjim95@gmail.com>

# Install OpenJDK-8
RUN   sudo apt-get update && \
      sudo apt-get install -y openjdk-8-jdk && \
      sudo apt-get install -y ant && \
      sudo apt-get clean;

# Fix certificate issues
RUN  sudo apt-get update && \
     sudo apt-get install ca-certificates-java && \
     sudo apt-get clean && \
     sudo update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME
# Show the installed version of JAVA
RUN java -version

# Install Wget
RUN sudo apt-get install wget &&  sudo apt-get clean;
RUN java -version

# Install MAVEN
RUN sudo apt-get install -y maven;
# Show the verison of Maven
RUN mvn -version

# Install and upgrade pip3 
RUN sudo apt-get install -y python3-pip;
RUN pip install --upgrade pip

# Install new version of pytorch
RUN pip3 install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install git
RUN sudo apt-get install -y git;

# Clone repositories
# UFSAC Repository
RUN git clone https://github.com/getalp/UFSAC.git && \
	 cd UFSAC/java && \
	 mvn install && \
	 cd ../../;

# Disambiguate Repository
RUN git clone https://github.com/getalp/disambiguate.git && \
	cd disambiguate/java && \
	mvn compile && \
	cd ../../;

