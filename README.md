# BATMAESTRO open-source implementation
BATMAESTRO is a battery model implementation in Python that includes battery lifetime modeling and that is compatible with mixed-integer linear optimization framework.

BATMAESTRO results from work funded by the SFOE.

BATMAESTRO was derived as a modification of the Open-SESAME-Battery is an open source battery lifetime simulation software, available [here](https://gitlab.ti.bfh.ch/oss/esrec/open-sesame) and developed by Bern University of Applied Science (BFH) with contributions of Centre Suisse d'Electronique et de Microtechnique (CSEM) within the framework of the IEA task 32 funded by SFOE.


## Purpose and context

Storage systems are an important enabler for the integration of increasing amounts of renewable energy. Electricity system planning and operation of electrochemical storage systems require models that are suitable for use within mathematical optimization that capture sufficiently accurately the main operating limitations of batteries, such as degradation resulting in a reduction of the available storage capacity. This project has led to the creation of an open-source model for batteries that is built upon a public simulation model supported by experimental data. This model allows to properly consider multiple factors contributing to battery degradation in optimization frameworks based on traceable data. The model has been validated on the one hand by demonstrating errors in degradation prediction of less than 5% over a full year compared to previously available and validated empirical models based. On the other hand, it was demonstrated in simulated scenarios that it can be used to improve the lifetime and economic performance of batteries in arbitrage and self-consumption scenarios. For example, when utilized to optimize the charging profile of a battery performing arbitrage on electricity prices, it was found that the battery life could be extended to 12 years compared to 5 years when using only standard charging rate and state of charge limits, bringing the net benefit (including an annualized cost for battery replacement) from -1.2k to +10kCHF/MWh/year.

A publication has been submitted for review to the IEEE ISGT 2023 conference and is awaiting review:
_Method to Embed Behavioral Battery Model in Predictive Energy Management Systems_, A. Sutter, T. T. Gorecki and S. S. Bhoir

This code aims to support the reproduction of the results of this publication. Howevr, it does not allow to reproduce the full results presetned in the paper, since they were obtained by incorporating this battery model within the NRGMAESTRO™ library. See [here](https://www.csem.ch/news/smart-energy-management-systems) for more information on CSEM energy management activities and NRGMAESTRO™

## Content

The src folder containts the actual battery model implementation. We refer to the inline documentation and to the publication for the implementation details
At thee root of the repository, the main.py script allows to execute optimization scenarios with three levels of battery model
- A simple state-of-health model
- A complex state-of-health model
- A complex state-of-health and state-of-resistance model

## Get started 

The code require Python3.8+.

- install required python packages:  
Run: `pip install -r requirements.txt`

- run main script
Run: `python main.py`


