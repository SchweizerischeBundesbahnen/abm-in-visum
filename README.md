
Activity-based model in PTV Visum
================

![Example Schedule](docs/schedule_example.png "Example schedule in Visum")

``ABM-in-Visum`` is a collaborative project initiated by [Swiss Federal Railways (SBB)](https://www.sbb.ch/) and [PTV Group](https://company.ptvgroup.com/).

It provides the core functionality to simulate an *activity-based travel demand model* within the [PTV Visum software](https://www.ptvgroup.com/en/solutions/products/ptv-visum/). In addition to the core functionality, user-specific model implementations (see ``model-contribs``) are given.

## Overview & history



## Functionality

The funcionality in this repository an be divided into the two components:
1) ``engines`` - the purpose of this component is having a central place for very general core functionality. For now, it contains:
   - ``location choice engine``:
   - ``simple discrete choice engine``:
2) ``model-contribs`` - this folder contains user-specific model contributions, which can used as example for different implementations of the core ``engines``: 
   - ``sbb-mobi-plans``:
   - ``ptv-simple-example``: to come...

All scripts rely on having a synthetic population including a person database (with socio-demographic attributes and household membership) as well as a location database (household residences and other activity locations). This repository does not provide the functionality to generate a synthetic population.


## Usage

The usage of the scripts available in this repository requires a license for the [PTV Visum software](https://www.ptvgroup.com/en/solutions/products/ptv-visum/). *PTV Visum* offers the efficient data structures that are necessary to manage, store and visualize activity-based travel demand (released in version 2020).

Within the *PTV Visum* software, the procedure sequences allows to execute to provided Python scripts. The following figure shows an example of a complete procedure sequence for a activity-based model (as provided in ``model-contribs/sbb-mobi-plans``):
![Procedure Sequence](docs/procedure_sequence_example.png "Example procedure sequence in Visum")


## Collaboration

We are very happy to include any motivated collaborator. Just `clone` or `fork` ABM-in-Visum and extend it by your own contributations (either in the ``engines`` or as your own ``model-contribs``). In the case of a `clone`, you are welcome to work on your own branch within this repository. Then, come back with a pull request on the branch ``main`` - it will be reviewed by the repository admins and provided to all the other users of the repository after the merge.

## Additional information & literature

- PTV Visum Webinar: [Activity Based Modelling with PTV Visum](https://www.youtube.com/watch?v=HvxDVKPmS-s)
- Scherr, W., Manser, P., Joshi, C., Frischknecht, N., & Métrailler, D. (2020). Towards agent-based travel demand simulation across all mobility choices – the role of balancing preferences and constraints. *European Journal of Transport and Infrastructure Research*, 20(4), 152-172. (https://doi.org/10.18757/ejtir.2020.20.4.4463).
- Scherr, W., Manser, P., Bützberger, P. (2020). SIMBA MOBI: Microscopic Mobility Simulation for Corporate Planning. *Transportation Research Procedia*, 49, 30-43. (https://doi.org/10.1016/j.trpro.2020.09.004).



-----

Licensing
=========
``ABM-in-Visum`` is licensed under the Apache License, Version 2.0. See
[LICENSE](https://github.com/SchweizerischeBundesbahnen/abm-in-visum/blob/main/LICENSE) for the full
license text.
