# AnimalGAN: A Generative Adversarial Network Model Alternative to Animal Studies for Clinical Pathology Assessment
**Pretrained Models**

- [AnimalGAN](https://drive.google.com/open?id=). Download and save it to `models/AnimalGAN`.
- [Structure-based splitting](https://drive.google.com/open?id=)
- [Approval year-based splitting](https://drive.google.com/open?id=)
- [ATC-based splitting](https://drive.google.com/open?id=)

**Evaluating**

- Run `python main.py --cfg cfg/AnimalGAN_eval.yml --gpu 2` to generate clinical pathology data for treatment conditions in test set.
- if you want to try other pretrained models, please replace AnimalGAN_eval.yml here
