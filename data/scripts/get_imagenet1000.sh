
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
d='../datasets/imagenet1000' 
mkdir -p $d && cd $d

# Download/unzip train
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/imagenet1000.zip
unzip imagenet1000.zip && rm imagenet1000.zip
