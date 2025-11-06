mkdir -p ./data
mkdir -p ./data/office31
mkdir -p ./data/officehome
cd ./data/office31
gdown https://drive.google.com/uc?id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE -O office31_images.tar.gz
tar -xzf office31_images.tar.gz
cd ../officehome
gdown 1FM7FAU8Q_CZaXnK95U4CEn52mEHFtNnm
unzip OfficeHomeDataset_10072016.zip 