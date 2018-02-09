This is the implementation for our S-HUNE graph embedding paper in MATLAB. It implements two algorithms:
- HUNE (High-order proximity-preserving Unique Node Embeddings): factorization-based graph embedding method
- SHUNE (Scalable HUNE): random-walk based graph embedding method

How to use (Tested on MATLAB 2014b, 2017a and 2017b):
- HUNE: 
1. run experiment_HUNE_embs.m

- SHUNE:
1. Compile shune.c using mex: mex shune.c
2. Run experiment_SHUNE_embs.m


- Evaluation on the node classification task (classification evaluation using Deepwalk testing code: you need python with gensim, sklearn and scipy installed)
1. run evaluation_node_classification.m (double check python sys.path to make sure python path is correct from matlab)
or from command line:
1. CURRENT_FOLDER/python ./scoring.py ./blogcatalog.mat ./embeddings_HUNE.mat ./classification_res_HUNE.mat











