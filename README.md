# Can LLMs Reason with Rules? Logic Scaffolding for Stress-Testing and Improving LLMs [[Paper]](https://arxiv.org/abs/2402.11442)

![Illustration of Logic Scaffolding](assets/logical_scaffolding.png)

This repository hosts the codes of our logic scaffolding inferential rule generation framework for primitive rule generation and rule composition, and the data of our generated inferential rule base ULogic. 

## Running the rule generation
You can directly run `primitive_rule_pipeline.ipynb` and `rule_composition.ipynb` respectively for primitive rule generation and rule composition in object interaction domain (for illustration). The functions (or step number and name) of each module in our scripts are clearly commentted before each section.

Before running, you need to set your API key for OpenAI. 

The codes for rule probing and our inference engine (training & evaluation) will be added.

## Dataset
`Data/ulogic.json` provides all inferential rules generated by our framework while `Data/probing_subset.json` provides a high-quality author verified subset of ULogic for rule probing of LLMs.
Each rule is formatted as following:
```json
{
   "s_rule": "CanObserve(Person X, Animal Y):- MoveTo(Person X, Region Z), Born(Animal Y, Region Z);",
   "v_rule": "If Person X moves to Region Z and Animal Y is born in Region Z, then Person X can observe Animal Y.",
   "domain": "accessibility",
   "depth": 0,
   "length": 2,
   "positive": true,
   "structure": "disjunctive",
   "label": true,
   "original_human_prediction": "2",
   "flipped_human_prediction": "1"
}
```

## Demo
We provide a [demo](http://210.16.188.56:59998) for our inference engine.

## Authors and Citation

This study was authored by Siyuan Wang, Zhongyu Wei, Yejin Choi and Xiang Ren. We encourage the use of our code and data in your research and kindly request citation of our paper as follows:

```BibTeX
@article{wang2024can,
  title={Can LLMs Reason with Rules? Logic Scaffolding for Stress-Testing and Improving LLMs},
  author={Wang, Siyuan and Wei, Zhongyu and Choi, Yejin and Ren, Xiang},
  journal={arXiv preprint arXiv:2402.11442},
  year={2024}
}
```
