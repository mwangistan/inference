### Running the benchmark on windows

git clone --recurse-submodules https://github.com/mwangistan/inference.git
cd inference/loadgen

pip install pycocotools, requests, boto3, tqdm, pandas
python setup.py develop

cd ../vision/classification_and_detection
python setup.py develop

#### Download openimages 
python tools\openimages.py -m <MAX_IMAGES> - d<DATA_SET_PATH>

### Running the benchmark
python python/main.py --profile retinanet-onnxruntime --model <Model path> --dataset-path <Path to openimages dataset> --accuracy
