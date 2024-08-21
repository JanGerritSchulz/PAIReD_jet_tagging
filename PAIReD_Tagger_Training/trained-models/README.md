# Pre-trained PAIReD taggers available here

Here, we list a number of available pre-trained PAIReD taggers. The models can be found in the respective directories here. They differ in their input features, output nodes and data trained on. All are based on the ParticleTransformer (ParT) architecture. They correspond to the networks trained in the Master's thesis *A first implementation of PAIReD jet tagging in the CMS experiment* by Jan Schulz, so more information can be found there.

| Tagger name                  | Inputs                             | Output nodes           | Processes used in network training |
|------------------------------|------------------------------------|------------------------|------------------------------------|
| [PAIReDEllipse 3 [DY]](./PAIReDEllipse%203%20[DY]/)         | PF candidates (excl. PUPPI weight) | LL, CC, BB             | VH(cc), VH(bb), DY+jj                |
| [PAIReDEllipse 3 SV [DY]](./PAIReDEllipse%203%20SV%20[DY]/)      | PF candidates + SVs                | LL, CC, BB             | VH(cc), VH(bb), DY+jj                |
| [PAIReDEllipse 3 SV [DY+W+tt]](./PAIReDEllipse%203%20SV%20[DY+W+tt]/) | PF candidates + SVs                | LL, CC, BB             | VH(cc), VH(bb), DY+jj, W+jj, tt     |
| [PAIReDEllipse 6 SV [DY+W+tt]](./PAIReDEllipse%206%20SV%20[DY+W+tt]/) | PF candidates + SVs                | ll, cl, bl, bb, CC, BB | VH(cc), VH(bb), DY+jj, W+jj, tt     |