cd home/ubuntu
if [ ! -d yolov5 ]; then
  echo "Running first-time script." #
  git clone https://github.com/ultralytics/yolov5 -b master && sudo chmod -R 777 yolov5
  cd yolov5
  bash data/scripts/get_coco.sh && echo "COCO done." &
  sudo docker pull ultralytics/yolov5:latest && echo "Docker done." &
  python -m pip install --upgrade pip && pip install -r requirements.txt && python detect.py && echo "Requirements done." &
  wait && echo "All tasks done." 
else
  echo "Running re-start script." 
  i=0
  list=$(sudo docker ps -qa) 
  while IFS= read -r id; do
    ((i++))
    echo "restarting container $i: $id"
    sudo docker start $id
    
    sudo docker exec -d $id python utils/aws/resume.py 
  done <<<"$list"
fi
