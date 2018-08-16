# lncRNAnet: Long Noncoding RNA Identification using Deep Learning
### DEPENDENCIES
[REQUIRED DEPENDENCIES]

The required dependencies are: 1)keras, 2) theano, 3) biopython 4) h5py

To install the dependencies, type the following command:
```
$ sudo pip3 install --upgrade -r requirements.txt
```
[PREREQUISITES]

Modify "image_data_format", "backend" from the keras configure file ('keras.json') as below:

The default keras configure file is at '~/.keras/keras.json'

```
{
    "epsilon": 1e-07,
    "image_data_format": "channels_first",
    "backend": "theano",
    "floatx": "float32"
}
```
### INPUTS
Fasta file


### USAGE
```
$ ./code/lncRNAnet.py input.fasta output.txt
```
