#sudo docker run --gpus all --name logAutoML --restart unless-stopped -p 25:22 tensorflow/tensorflow:latest-gpu
#sudo docker run --gpus all --name logAutoML -p 25:22 -it tensorflow/tensorflow:latest-gpu /bin/bash

#sudo docker exec -it logAutoML /bin/bash
#sudo docker stop logAutoML && sudo docker rm logAutoML

ssh log@192.168.0.25

# CREATE
sudo docker run --gpus all --name logAutoML -p 9999:8888 -it tensorflow/tensorflow:latest-gpu-py3-jupyter /bin/bash

# INIT
pip install --upgrade pip
pip install --upgrade autokeras
jupyter notebook --ip=0.0.0.0 --allow-root

# RE-ACCESS
sudo docker start logAutoML
sudo docker exec -it logAutoML /bin/bash
