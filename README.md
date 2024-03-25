# Finetuning LLaMA2 7B, Phi2, and Mistral on the Alpaca Dataset Using Unsloth and Accelerate

## Installation

Install the dependencies for your enviroment is already present in the files, however, if you wish to manually install them:

```sh
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install torch
pip install -q -U transformers datasets accelerate peft trl bitsandbytes
```

If using GPUs like Ampere, Hopper GPUs (RTX 30xx, RTX 40xx, A100, H100, L40)...

```sh
pip install --no-deps packaging ninja einops flash-attn 
```


## Usage
The provided `master_file.ipynb` file contains the necessary documentation and commands to run the code. However, you must have access to a GPU with approximately 24GB of VRAM. If you wish to use the model on a GPU with less VRAM, it is possible but you may need to adjust the `dype=float32` values with `dtype=float16`.

## Results
At first, I evaluated the model using the default settings found using the HuggingFace generate function and achieved the following results.

Model      | Bleu Score | Rouge Score  | BertF1 | Human
| ------ | ------ | ------ | ------ | -- |
LLaMA2 7B  | 0.00000    | 0.62821      | 0.80228   | 0.86
Mistral    | 0.00000    | 0.63366      | 0.79927   |0.89
Phi2       | 0.00000    | 0.63741      | 0.80384| 0.90

The computer evaluated models overall don't reveal too much. The Bleu score is effectively useless due to the low score it achieves, the BertF1 score is fairly high and has little varience. The Rouge score here appears to also have this effect, but it shows more variation when modifying the parameters. The human scoring I think was better because we focused more on the response section of the output instead of the entire piece of text. I do personally believe that weighing the correct response should be higher than equal to grammar.

### Temperature Variations Tests

Temperature: 0.001 | Bleu Score | Rouge Score  | BertF1
|-|-|-|-|
Model      |  |   |     
LLaMA2 7B  | 0.00000    | 0.63347      | 0.80563   
Mistral    | 0.00000    | 0.63210      | 0.79742   
Phi2       | 0.00000    | 0.63802      | 0.80226   

Temperature: 0.25 | Bleu Score | Rouge Score  | BertF1
|-|-|-|-|
Model      |  |   |     
LLaMA2 7B  | 0.00000    | 0.62554      | 0.79959   
Mistral    | 0.00000    | 0.63716      | 0.79905   
Phi2       | 0.00000    | 0.64338      | 0.80494  

Temperature: 0.5 | Bleu Score | Rouge Score  | BertF1
|-|-|-|-|
Model      |  |   |     
LLaMA2 7B  | 0.00000    | 0.63729      | 0.80476   
Mistral    | 0.00000    | 0.61920      | 0.79508   
Phi2       | 0.00000    | 0.63706      | 0.80173  

Temperature: 0.75 | Bleu Score | Rouge Score  | BertF1
|-|-|-|-|
Model      |  |   |     
LLaMA2 7B  | 0.00000    | 0.61024      | 0.78862   
Mistral    | 0.00000    | 0.62698      | 0.79942   
Phi2       | 0.00000    | 0.62472      | 0.79878

### Beam Size Variations Tests

Beam Size: 1      | Bleu Score | Rouge Score  | BertF1    
|-|-|-|-|
Model      |  |   |     
LLaMA2 7B  | 0.00000    | 0.62731      | 0.79732   
Mistral    | 0.00000    | 0.60719      | 0.79311   
Phi2       | 0.00000    | 0.59868      | 0.77311  

Beam Size: 2     | Bleu Score | Rouge Score  | BertF1    
|-|-|-|-|
Model      |  |   |     
LLaMA2 7B  | 0.00000    | 0.63415      | 0.79749   
Mistral    | 0.00000    | 0.63944      | 0.79521   
Phi2       | 0.00000    | 0.64762      | 0.80356 

Beam Size: 4      | Bleu Score | Rouge Score  | BertF1    
|-|-|-|-|
Model      |  |   |     
LLaMA2 7B  | 0.00000    | 0.63459      | 0.79710   
Mistral    | 0.00000    | 0.63602      | 0.79331   
Phi2       | 0.00000    | 0.64330      | 0.80417  

Beam Size: 8     | Bleu Score | Rouge Score  | BertF1    
|-|-|-|-|
Model      |  |   |     
LLaMA2 7B  | 0.00000    | 0.62951      | 0.79618   
Mistral    | 0.00000    | 0.62115      | 0.79301   
Phi2       | 0.00000    | 0.64089      | 0.80295
### Top K Variations Tests
Top K:-2      | Bleu Score | Rouge Score  | BertF1    
|-|-|-|-|
Model      |  |   |     
LLaMA2 7B  | 0.00000    | 0.64563      | 0.80560   
Mistral    | 0.00000    | 0.60735      | 0.78687   
Phi2       | 0.00000    | 0.63009      | 0.79748 

Top-K: 4      | Bleu Score | Rouge Score  | BertF1    
|-|-|-|-|
Model      |  |   |     
LLaMA2 7B  | 0.00000    | 0.63535      | 0.80471   
Mistral    | 0.00000    | 0.62122      | 0.79976   
Phi2       | 0.00000    | 0.61698      | 0.79038 

Top-K: 8      | Bleu Score | Rouge Score  | BertF1    
|-|-|-|-|
Model      |  |   |     
LLaMA2 7B  | 0.00000    | 0.62703      | 0.79655   
Mistral    | 0.00000    | 0.59929      | 0.78640   
Phi2       | 0.00000    | 0.61810      | 0.78711   

Top-K:  16     | Bleu Score | Rouge Score  | BertF1    
|-|-|-|-|
Model      |  |   |     
LLaMA2 7B  | 0.00000    | 0.62263      | 0.79828   
Mistral    | 0.00000    | 0.61639      | 0.79472   
Phi2       | 0.00000    | 0.60339      | 0.78556

The main three hyperparameters are compared above. The general trend was that the temperature increasing caused the models to behave more eradically and generated irrelavant information, this is to be expected when increasing the temperature. Increasing the beam size had a positive effect and all of the metrics increased as it did. This is also expected since it controls the number of candidate paths taken at each step. The top-k tests showed that the score decreased as the top-k increases. This controls the number of vocabulary words the model can choose from during generation, so the more words there can sometimes be more fluency but a potential to stray from the truth.

## References
Unsloth Documentation
HuggingFace
## License
MIT License

Copyright (c) 2024 ouroborosrex

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
