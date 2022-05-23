FILE_LIST=$(gsutil ls gs://gcp-public-data--gnomad/release/3.1.2/vcf/genomes/*sites*bgz)
for GS_FILE in ${FILE_LIST}
do
    echo $GS_FILE
    gsutil -m cp $GS_FILE ./
    gunzip -c $(basename $GS_FILE) | cut -f 1,2,3,4,5 | gsutil cp - gs://korvaz/mpra_model_manuscript/data/gnomAD/$(basename $GS_FILE | sed 's/.bgz//')
    rm $(basename $GS_FILE)
done