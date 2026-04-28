#  Start  from  a  Python  3.10  Debian-based  image
FROM python:3.10-slim-bookworm

#  Set  user  to  root  for  system  package  and  n8n  installation
USER root

#  Install  system  dependencies
RUN apt-get  update  &&  apt-get  install  -y  --no-install-recommends  tini  npm  curl  &&  curl  -fsSL  https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*


#  Install  n8n  globally  using  npm
RUN npm  install  -g  n8n

#  Create  a  directory  for  our  app  and  scripts
WORKDIR /app

#  Copy  requirements  file  and  install  Python  packages
COPY requirements.txt  .
RUN pip  install  -r  requirements.txt

#  Create  a  non-root  user  and  directory  for  n8n's  own  data
RUN adduser  --system  --group  --home  /home/node  node

#  Expose  n8n  port
EXPOSE 5678

#  Switch  to  the  n8n  user
USER node
WORKDIR /home/node

#  Define  the  entrypoint  to  run  n8n
ENTRYPOINT ["tini",  "--",  "n8n"]
