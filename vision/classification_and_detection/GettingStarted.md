## Running the benchmark on windows

### Clone the repo
```
git clone --recurse-submodules https://github.com/mwangistan/inference.git
cd inference/loadgen
```

### Setup loadgen
```
cd inference/loadgen
python setup.py develop
```

### Setup the vision app
```
cd ../vision/classification_and_detection
python setup.py develop
```
### Download openimages 
```
python tools\openimages.py --max-images <MAX_IMAGES> --dataset-dir<DATA_SET_PATH>
```

### Run the benchmark
```
python python/main.py --profile <Profile> --model <Model path> --dataset-path <Path to openimages dataset> --accuracy
```
