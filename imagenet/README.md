# TTAC on ImageNet

TTAC on ImageNet under common corruptions.

### Requirements

- To install requirements:

    ```
    pip install -r requirements.txt
    ```

- To download dataset:

    We need to firstly download the validation set and the development kit (Task 1 & 2) of ImageNet-1k on [here](https://image-net.org/challenges/LSVRC/2012/index.php), and put them under `data` folder.

    The structure of the `data` folder should be like

    ```
    data
    |_ ILSVRC2012_devkit_t12.tar
    |_ ILSVRC2012_img_val.tar
    ```

- To create the corruption dataset
    ```
    python utils/create_corruption_dataset.py
    ```

    The issue `Frost missing after pip install` can be solved following [here](https://github.com/hendrycks/robustness/issues/4#issuecomment-427226016).

    Finally, the structure of the `data` folder should be like
    ```
    data
    |_ ILSVRC2012_devkit_t12.tar
    |_ ILSVRC2012_img_val.tar
    |_ val
        |_ n01440764
        |_ ...
    |_ corruption
        |_ brightness.pth
        |_ contrast.pth
        |_ ...
    |_ meta.bin
    ```

### Pre-trained Models

Here, we use the pretrain model provided by torchvision.

### Results

We mainly conduct our experiments under the sTTT (N-O) protocol, which is more realistic and challenging.

- run TTAC on ImageNet-C under the sTTT (N-O) protocol.
  
    ```
    bash scripts/run_ttac_no.sh
    ```

    The following results are yielded by the above script (classification errors) under the snow corruption:

    | Method | ImageNet-C (Level 5) |
    |:------:|:----------:|
    |  Test  |   82.22    |
    |  TTAC  |   44.56     |

- run TTAC on ImageNet-C under the N-O without queue protocol.
  
    In the sTTT protocol, we employ a sample queue, storing past samples, to aid model adaptation to enhance stability and improve accuracy. Obviously, it would bring more computing cost. 

    Therefore, we provide the version of TTAC without queue for more efficiency.

    ```
    bash scripts/run_ttac_no_without_queue.sh
    ```

    The following results are yielded by the above script (classification errors) under the snow corruption:

    | Method | ImageNet-C (Level 5) |
    |:------:|:----------:|
    |  Test  |   82.22    |
    |  TTAC  |   46.64     |
