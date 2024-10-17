wget -O UCI.zip https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip

mkdir -p UCI

unzip UCI.zip -d UCI/

unzip "UCI/UCI HAR Dataset.zip" -d UCI/

rm UCI.zip

rm -r "UCI/__MACOSX/"
rm -r "UCI/UCI HAR Dataset.names"
rm -r "UCI/UCI HAR Dataset.zip"

python setupDatasets.py