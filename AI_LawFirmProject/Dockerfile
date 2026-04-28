#  Start  from  a  Python  3.10  Debian-based  image
FROM python:3.10-slim-bookworm

#  Set  user  to  root  for  system  package  and  n8n  installation
USER root

#  Install  system  dependencies  using  NodeSource  GPG  key  (avoids  piping  remote  scripts  to  bash)
RUN apt-get  update  &&  apt-get  install  -y  --no-install-recommends  tini  curl  ca-certificates  gnupg  \
    &&  mkdir  -p  /etc/apt/keyrings  \
    &&  curl  -fsSL  https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key  \
        |  gpg  --dearmor  -o  /etc/apt/keyrings/nodesource.gpg  \
    &&  echo  "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main"  \
        >  /etc/apt/sources.list.d/nodesource.list  \
    &&  apt-get  update  &&  apt-get  install  -y  --no-install-recommends  nodejs  \
    &&  rm  -rf  /var/lib/apt/lists/*


#  Install  n8n  globally  using  npm
RUN npm  install  -g  n8n

#  Create  a  directory  for  our  app  and  scripts
WORKDIR /app

#  Copy  requirements  file  and  install  Python  packages
COPY requirements.txt  .
RUN pip  install  --no-cache-dir  -r  requirements.txt

#  Create  a  non-root  user  and  directory  for  n8n's  own  data
RUN adduser  --system  --group  --home  /home/node  node

#  Expose  n8n  port
EXPOSE 5678

#  Switch  to  the  n8n  user
USER node
WORKDIR /home/node

#  Define  the  entrypoint  to  run  n8n
ENTRYPOINT ["tini",  "--",  "n8n"]
