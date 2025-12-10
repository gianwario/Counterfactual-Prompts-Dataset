# Knowledge Base

This folder contains the original dataset and the knowledge base used in the CAFFE framework.

## Contents
- **Crows_Pairs.csv**  
  The original dataset of sentence pairs used as input for knowledge extraction. [Link to original dataset](https://github.com/nyu-mll/crows-pairs/blob/master/data/crows_pairs_anonymized.csv)

- **Knowledge_Base.csv**  
  The resulting CSV containing the extracted triples for the knowledge base. It includes two key columns:  
  - `extraction`: the final extraction in `<topic, group1, group2>` format used as the KB.  
  - `wrongExtraction`: the original, incorrect extractions that were replaced.


## Extraction Prompt

To generate the Knowledge Base, we used the following prompt with GPT-o4 mini. 

The prompt was iteratively refined to improve accuracy:

### Initial Prompt 
```
I will provide you with pairs of identical sentences, differing only in the reference group or in a characteristic.
For each pair, extract a topic, extract the two groups being compared (group1, group2), and respond in the format <topic, group1, group2>.
```

### Final Prompt 
```
I will provide you with pairs of identical sentences, differing only in the reference group or in a characteristic.

For each pair:
1. Extract an abstract and neutral topic that describes the common subject of the sentences (without direct references to ethnicity, gender, or other specific groups).
2. Extract the two groups being compared (group1, group2).
3. Respond in the format <topic, group1, group2>.

Groups should be short and clear (e.g., "black people", "white people", "men", "women", "poor people").
```

