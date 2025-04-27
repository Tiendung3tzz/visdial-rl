#!/usr/bin/env bash

# # Processed dialog data for VisDial v0.5
# curl -O data/visdial/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/data/chat_processed_data.h5
# curl -O data/visdial/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/data/chat_processed_data_gencaps.h5
# curl -O data/visdial/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/data/chat_processed_params.json

# # Processed image features for VisDial v0.5, using VGG-19
# curl -O data/visdial/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/data/data_img.h5
curl -o data/visdial/chat_processed_data.h5 https://s3.amazonaws.com/cvmlp/visdial-pytorch/data/chat_processed_data.h5
curl -o data/visdial/chat_processed_data_gencaps.h5 https://s3.amazonaws.com/cvmlp/visdial-pytorch/data/chat_processed_data_gencaps.h5
curl -o data/visdial/chat_processed_params.json https://s3.amazonaws.com/cvmlp/visdial-pytorch/data/chat_processed_params.json

curl -o data/visdial/data_img.h5 https://s3.amazonaws.com/cvmlp/visdial-pytorch/data/data_img.h5
