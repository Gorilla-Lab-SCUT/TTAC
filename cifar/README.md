# TTAC on CIFAR10/100

TTAC on CIFAR-10/100 under common corruptions or natural shifts. Our implementation is based on [repo](https://github.com/vita-epfl/ttt-plus-plus/tree/main/cifar) and therefore requires some similar preparation processes.


### Requirements

To install requirements:

```
pip install -r requirements.txt
```

To download datasets:

```
export DATADIR=/data/cifar
mkdir -p ${DATADIR} && cd ${DATADIR}
wget -O CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf CIFAR-10-C.tar
wget -O CIFAR-100-C.tar https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1
tar -xvf CIFAR-100-C.tar
```

### Pre-trained Models

The checkpoints of pre-train Resnet-50 can be downloaded (214MB) using the following command:

```
mkdir -p results/cifar10_joint_resnet50 && cd results/cifar10_joint_resnet50
gdown https://drive.google.com/uc?id=1TWiFJY_q5uKvNr9x3Z4CiK2w9Giqk9Dx && cd ../..
mkdir -p results/cifar100_joint_resnet50 && cd results/cifar100_joint_resnet50
gdown https://drive.google.com/uc?id=1-8KNUXXVzJIPvao-GxMp2DiArYU9NBRs && cd ../..
```

These models are obtained by training on the clean CIFAR10/100 images using semi-supervised SimCLR.

### One Pass Protocols:

- run TTAC on CIFAR10\100-C under the sTTT (N-O) protocol.

    ```
    # CIFAR10-C: 
    bash scripts/run_ttac_cifar10_no.sh

    # CIFAR100-C: 
    bash scripts/run_ttac_cifar100_no.sh
    ```

    The following results are yielded by the above scripts (classification errors) under the snow corruption:


    | Method | CIFAR10-C | CIFAR100-C |
    |:------:|:---------:|:----------:|
    |  Test  |   21.93   |    54.57   |
    |  TTAC  |   10.01   |    37.69   |

- run TTAC on CIFAR10\100-C under the N-O without queue protocol.
    
    In the sTTT protocol, we employ a sample queue (for all comparing methods), storing past samples, to aid model adaptation to enhance stability and improve accuracy. Obviously, it would bring more computing cost. 
    
    Therefore, we provide the version of TTAC without queue which can be utilized in cases where efficiency is important.

    ```
    # CIFAR10-C: 
    bash scripts/run_ttac_cifar10_no_without_queue.sh

    # CIFAR100-C: 
    bash scripts/run_ttac_cifar100_no_without_queue.sh
    ```

    The following results are yielded by the above scripts (classification errors) under the snow corruption:

    | Method | CIFAR10-C | CIFAR100-C |
    |:------:|:---------:|:----------:|
    |  Test  |   21.93   |    54.57   |
    |  TTAC  |   11.91    |    40.39   |


- run TTAC on CIFAR10\100-C under the Y-O protocol.

    ```
    # CIFAR10-C: 
    bash scripts/run_ttac_cifar10_yo.sh

    # CIFAR100-C: 
    bash scripts/run_ttac_cifar100_yo.sh
    ```

    The following results are yielded by the above scripts (classification errors) under the snow corruption:

    | Method | CIFAR10-C | CIFAR100-C |
    |:------:|:---------:|:----------:|
    |  Test  |   21.93   |    54.57   |
    |  TTAC  |   9.99    |    34.97   |

### Multiple Pass Protocols:

- run TTAC on CIFAR10\100-C under the N-M protocol.

    ```
    # CIFAR10-C: 
    bash scripts/run_ttac_cifar10_nm.sh

    # CIFAR100-C: 
    bash scripts/run_ttac_cifar100_nm.sh
    ```

    The following results are yielded by the above scripts (classification errors) under the snow corruption:

    | Method | CIFAR10-C | CIFAR100-C |
    |:------:|:---------:|:----------:|
    |  Test  |   21.93   |    54.57   |
    |  TTAC  |   8.80    |    34.29   |

- run TTAC on CIFAR10\100-C under the Y-M protocol.

    ```
    # CIFAR10-C: 
    bash scripts/run_ttac_cifar10_ym.sh

    # CIFAR100-C: 
    bash scripts/run_ttac_cifar100_ym.sh
    ```

    The following results are yielded by the above scripts (classification errors) under the snow corruption:

    | Method | CIFAR10-C | CIFAR100-C |
    |:------:|:---------:|:----------:|
    |  Test  |   21.93   |    54.57   |
    |  TTAC  |   8.00    |    30.48   |


### Descriptions

- Both `TTAC_multipass.py` and `TTAC_multipass2.py` are the implementations of TTAC under multiple pass protocol, except that `TTAC_multipass.py` will be more memory efficient but slower, while `TTAC_multipass2.py` will be faster. 
- `TTAC_onepass.py` and `TTAC_onepass2.py` are similar situation. 


### Acknowledgements

Our code is built upon the public code of the [TTT++](https://github.com/vita-epfl/ttt-plus-plus/tree/main/cifar).