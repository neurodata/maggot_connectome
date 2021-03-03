# Data
Currently, the data used for this project is released as the paper presenting the work
has not been published. Hopefully, data will be publicly available by the summer of 
2021.


## Cell type classes
WARNING: the following are cell types, classes, or otherwise groupings of neurons that 
do not necessarily correspond exactly to columns in the current metadata. Ideally we'll
improve this documentation so that every column in the metadata is one-to-one with an
explanation.

Sensory
---
- `sens`: anything labeled as a sensory neuron
   - Usually just get the axon entering the volume in this dataset
   - There are several nerve bundles coming in
   - Subclasses:
      - `ORN`: odorant receptor
         - Paper: Berck et al.
         - These are in the antennal nerve also
      - `AN`: antennal nerve (anatomical)
         - Paper: https://elifesciences.org/articles/40247
         - In the antennal nerve but not odor receptors
         - Pharyngeal/internal organ info
      - `MN`: maxillary nerve (anatomical)
         - Paper: https://elifesciences.org/articles/40247
         - Gustatory/somatosensory
      - `PaN`: prothoracic accessory nerve (anatomical)
         - Paper: https://elifesciences.org/articles/40247
         - Gustatory/somatosensory
      - `Photo`: photoreceptor
         - Paper: https://elifesciences.org/articles/28387
      - `Temp`: temperature sensing
         - Paper: no paper
         - Come in as part of AN (I think)

Output 
--- 
- All of these will have the `O_` tag at the beginning
- `dVNC`: descending to ventral nerve cord
- `dSEZ`: descending to sub-esophageal zone
- `RG`: ring gland
   - Hormone outputting organ, located near top of the brain
   - Subclasses: 
      - `IPC`: insulin producing cells
         - Paper: https://elifesciences.org/articles/16799
      - `ITP`
      - `CA-LP`
      - `CRZ` 
      - `DMS`
- Motor neurons in SEZ

Mushroom body
--- 
- Clearly anatomically defined region (for the most part)
- Paper: [Eichler et al. 2017](https://www.nature.com/articles/nature23455)
- `KC`: Kenyon cells
   - Subclasses:
      - `-{claw number}`
- `MBIN`: Mushroom body input neuron
- `MBON`: Mushroom body output neuron

Antennal lobe
--- 
- Anatomical region
- Paper: [Berck et al. 2016](https://elifesciences.org/articles/14859)
- `bLN`
   - `Trio`
   - `Duet`
- `cLN`
   - `choosy`
   - `picky`
- `keystone`

Lateral horn and convergence neurons
---
- No paper
- Defined based on normalized input thresholding

Feedback neurons
---
- Claire's paper