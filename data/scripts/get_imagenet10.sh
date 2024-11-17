
if [ "$#" -gt 0 ]; then
  for opt in "$@"; do
    case "${opt}" in
    --train) train=true ;;
    --val) val=true ;;
    esac
  done
else
  train=true
  val=true
fi

# Make dir
d='../datasets/imagenet10' 
mkdir -p $d && cd $d

# Download/unzip train
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/imagenet10.zip
unzip imagenet10.zip && rm imagenet10.zip
