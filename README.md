# ViRDA: Reusing Backbone for Unsupervised Domain Adaptation with Visual Reprogramming

## 📒 Abstract

Image classification is the foundation of nearly all computer-vision pipelines. While state-of-the-art models excel within their training domains, their performance often deteriorates when transferred to a new, unlabeled setting. Unsupervised domain adaptation (UDA) tackles this challenge by repurposing a well-trained source classifier for a target domain, allowing strong downstream results without the need for additional labeled data. Existing UDA pipelines fine-tune already well-trained backbone parameters for every new source-and-target pair, making the number of training parameters and storage memory grow linearly with each new pair, and also preventing the reuse of these well-trained backbones’ parameters.

Inspired by recent implications that existing backbones have textural biases, we propose making use of domain-specific textural bias for domain adaptation via visual reprogramming, namely VirDA. Instead of fine-tuning the full backbone, VirDA prepends a domain-specific visual reprogramming layer to the backbone. This layer produces visual prompts that act as an added textural bias to the input image, adapting its “style” to a target domain. To optimize these visual reprogramming layers, we use multiple objective functions that optimize the intra- and inter-domain distribution differences when domain-adapting visual prompts are applied. This process does not require modifying the backbone parameters, allowing the same backbone to be reused across different domains.

We evaluate VirDA on the Office-31 dataset, where VirDA achieves 86.80% mean accuracy. In comparison with FixBi, the state-of-the-art UDA method that uses the same backbone, VirDA requires only 4.04% of the amount of parameters while exhibiting less than 5% performance drop. Furthermore, on both Office-31 and Office-Home, VirDA consistently outperforms MSTN and DANN while using only 3.56% of the amount of trainable parameters. We release VirDA’s source code tentatively.

![Main model](assets/model.png)

## 💡 Preparation

Run these installations to download the datasets: Office-Home and Office-31. The Digits dataset is included within the Torchvision library.

```bash
bash datasets.sh
````

## 🔥 Get Start

Train the model from scratch with the default settings:

```bash
python train.py --config=configs/office_31/a2d.yaml
```

Train the model for domain adaptation starting from a well-trained source model:

```bash
python domain_adapt.py --config=configs/office_31/a2d.yaml --ckpt=path/to/checkpoints
```

Evaluate the model:

```bash
python eval.py --config=configs/office_31/a2d.yaml --ckpt=path/to/checkpoints
```

## 📦 Well-Trained Models

We saved our checkpoints on HuggingFace at this [repo](https://huggingface.co/G7xHp2Qv/ViRDA) and thus can be downloaded via the prepared script. For each task, we provide both the burn-in and domain-adaptation best checkpoints.

```bash
python download_all_checkpoints.py
```