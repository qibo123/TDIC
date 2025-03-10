# TDIC
The goal of this project is to model user interest and conformity in a time-aware manner. TDIC disentangles interest and conformity using item popularity.


## Environment
* Python = 3.8.13
* Pytorch = 1.12.0
* numpy = 1.22.4
* tqdm = 4.64.0
### Train & Test
Train and evaluate the model with the following commands.
You can also add command parameters to specify dataset/GPU id, or get fast execution by dataset sampling.

```shell
cd src
# --dataset: (str) dataset name
# --msl: (int) max sequence length
# --gpu: (int) gpu id
# --sample: (int) sample number for fast execution
python run_clhhn.py
```
