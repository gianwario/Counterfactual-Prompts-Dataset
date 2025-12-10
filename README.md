# Counterfactual Prompt Pairs Dataset Card

## Dataset Summary

This dataset contains **counterfactual prompt pairs** designed to evaluate **bias, robustness, and fairness** in Large Language Models (LLMs).
For each combination of **bias type**, **intent**, and **semantic topic**, the dataset provides **20 aligned counterfactual pairs** (40 sentences), each pair containing prompts identical except for the **sensitive group attribute**.

The dataset supports research on (not limited to):

- Stereotype and bias detection for specific communicative intents
- Counterfactual fairness testing
- Sensitivity of LLM outputs to protected attributes

---

## Repository Structure

```
/
├── data/
│   ├── 1_raw_dataset.csv ## The raw csv datasets containing separated prompts
│   ├── dataset.jsonl ## The processed jsonl file containing the dataset with grouped counterfactual prompts
│   └── stats_summary.json ## A jsonl file containing descriptive stats of the dataset
├── knowledge_base/ ## A folder containing the knowledge base from which the topics and sensitive groups have been extracted (specific details are in the folder)
├── scripts/
│   ├── 1_analyze_raw_dataset.py ## The script used to check consistency and gather descriptive statistics from the raw csv dataset
│   ├── 2_create_formatted_dataset.py ## The script used to create the formatted jsonl final dataset
│   ├── 3_example_usage.py ## A script containing an example usage of the dataset alongside results for two example runs
└── README.md
```

---

## Dataset Structure

### Raw CSV Format

| Column        | Description                                              |
| ------------- | -------------------------------------------------------- |
| `topic`     | Semantic theme (e.g.,*access to education*)            |
| `intent`    | Communicative intent (*Question, Statement*)           |
| `group`     | Sensitive attribute group (*black kids, white kids*)   |
| `sentence`  | The prompt text                                          |
| `bias_type` | Category of sensitive attribute (*race-color, gender*) |

---

### Processed JSONL Format

Each line contains one complete counterfactual pair:

```json
{
  "id": "race-color||Question||access to quality education||0",
  "bias_type": "race-color",
  "intent": "Question",
  "topic": "access to quality education",
  "pair_index": 0,
  "groups": ["black kids", "white kids"],
  "prompts": [
    {"group": "black kids", "sentence": "..."},
    {"group": "white kids", "sentence": "..."}
  ]
}
```

---

## Sensitive Attributes Covered

- **Race / ethnicity**
- **Gender**
- **Nationality**
- **Age groups**
- **Socioeconomic categories**
- **Disability**
- **Sexual orientation**
- **Physical appearence**
- **Religion**

---

## Limitations

- Contains prompts involving sensitive attributes; handle responsibly.
- Some counterfactual swaps may not be culturally symmetric.

---

## Ethical Considerations

- Models tested with this dataset may produce biased or harmful responses.
- Prompts should not be used for real-world decision-making systems.
- Outputs should be analyzed with care to avoid reinforcing stereotypes.

---
