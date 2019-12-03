CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

export PYTHONPATH=$PYTHONPATH:$CODE_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

pip install -r $CODE_DIR/requirements.txt

# Create the necessary directories
DATA_DIR=data
OUTPUT_DIR=data/output_data
PLOTS_DIR=plots
LOGS_DIR=logs
TEST_LOGS_DIR=../tests/logs

mkdir -p $DATA_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $PLOTS_DIR
mkdir -p $LOGS_DIR
mkdir -p $TEST_LOGS_DIR

# Enter your company name and download the DICOM data and contour files
read -p 'Enter your company name: ' CO
CO=${CO,,}
DATA_LINK='https://s3.amazonaws.com/'$CO'-vision/coding_challenge/final_data.tar.gz'
wget $DATA_LINK -P $DATA_DIR
tar -xzf $DATA_DIR/final_data.tar.gz -C $DATA_DIR
rm $DATA_DIR/final_data.tar.gz