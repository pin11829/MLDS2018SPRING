## Training
copy the dataset to `data/clr_conversation.txt` then run
```
bash train.sh
```
## Testing 
```
bash hw2_seq2seq.sh <input file> <output file>
```
## Result
|                 |Greedy| Beam|
|-----------------|------|-----|
|Perplexity       | 11.02|10.52|
|Correlation score|   0.7| 0.67|
