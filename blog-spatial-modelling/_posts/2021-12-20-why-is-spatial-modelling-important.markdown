---
layout: post
title:  "Why Spatial Modeling is Important for Sustainable Development?"
date:   2021-12-20 12:30:32 +0800
category: Spatial Modeling
---

[The 2030 Agenda for Sustainable Development](https://sdgs.un.org/2030agenda), adopted by all United Nations Member States in 2015,
provides a shared blueprint for peace and prosperity for our people and our planet. One of its vision is:
>we envisage a world in which every country enjoys sustained, inclusive and sustainable economic growth and decent work for all. A world in which consumption and production patterns and use of all natural resources – from air to land, from rivers, lakes and aquifers to oceans and seas - are sustainable. One in which democracy, good governance and the rule of law as well as an enabling environment at national and international levels, are essential for sustainable development, including sustained and inclusive economic growth, social development, environmental protection and the eradication of poverty and hunger. One in which development and the application of technology are climate-sensitive, respect biodiversity and are resilient. One in which humanity lives in harmony with nature and in which wildlife and other living species are protected. This sounds fantastic! But some may doubt whether and when we can get there and how to get there? Frankly, I have no idea as well. Some may think this is too good to be true and therefore do nothing. Some may think this is irrelevant to them and therefore do nothing. However, I do know that a lot of people have been working on this no matter what. 

<iframe width="100%" height="360" src="https://www.youtube.com/embed/A9iRBVEU72c" frameborder="0" allowfullscreen></iframe>

Well then, to achieve that goal or dream, as data scientist, what can **we** do? This is my motivation to start this blog site. 
We want to do a little thing that may make this world a little better. Right? To get there, all of us need to work together. 

<iframe width="100%" height="360" src="https://www.youtube.com/embed/pvuN_WvF1to" frameborder="0" allowfullscreen></iframe>

---
<br/>

### What is Spatial Modeling?

Our environment, economy, and society are facing a lot of challenges.
Spatial modeling can help us better understand these environmental or social phenomena with mathematical and statistical models. 

The objective of spatial modeling is to study and simulate phenomena that occur in the real world (or something occurs in the Metaverse or simulate the phenomenon in Metaverse)
and facilitate data-driven problem solving and planning. For example, spatial modeling can be used to
analyze the projected path of tornadoes [(here)](https://www.noaa.gov/news/december-2021-tornado-outbreak-explained).
With spatial modeling, we can analyze the spatial temporal pattern of the "breath" and provide data-driven
insights to facilitate policy decision-making [(here)](https://spacetimewithr.org/).
We can watch the Earth breathe (averaged atmospheric carbon dioxide) from space based on earth observation data.
<iframe width="100%"  height="360" src="https://www.youtube.com/embed/NMbsszZ6zhc" frameborder="0" allowfullscreen></iframe>

Apart from natural phenomena, the book [Spatio-Temporal Statistics with R][1]{:target="_blank"} has considered 
other spatial temporal phenomena and events such as "describe the growth and decline of populations,
the territorial expansion and contraction of empires, the spread of world religions,
species (including human) migrations, the dynamics of epidemics, and so on". As it is said in the book, 
> Indeed, history and geography are inseparable. From this “big picture” point of view,
> there is a complex system of interacting physical, biological, and social processes across a range of spatial/temporal scales.

It is worth noting that spatial modeling here stands for spatio-temporal modeling, as
> it is not enough to consider just spatial snapshots of a process at a given time, nor just time-series profiles at a given spatial location – the behavior at spatial locations at one time point will almost certainly affect the behavior at nearby patial locations at the next time point. Only by considering time and space together can we address how spatially coherent entities change over time or, in some cases, why they change. It turns out that a big part of the how and why of such change is due to interactions across space and time, and across multiple processes.

---

### Why is Spatial Modeling Important for Sustainable Development?
The book [Spatio-Temporal Statistics with R][1]{:target="_blank"} gave an example about using spatial modeling to detect El Niño and La Niña events in the tropical Pacific Ocean.
> El Niño and La Niña phenomena correspond to periods of warmer-than-normal and colder-than-normal sea surface temperatures (SST), respectively. These SST “events” occur every two to seven years, although the exact timing of their appearance and their end is not regular. But it is well known that they have a tremendous impact on the weather across the globe, and weather affects a great number of things! For example, the El Niño and La Niña events can affect the temperature and rainfall over the midwest USA, which can affect, say, the soil moisture in the state of Iowa, which would likely affect corn production and could lead to a stressed USA agro-economy during that period. Simultaneously, these El Niño and La Niña events can also affect the probability of tornado outbreaks in the famed “tornado alley” region of the central USA, and they can even affect the breeding populations of waterfowl in the USA.

Associate with the recent tornadoes that devastated kentucky and four other states, we can understand that accurate national weather service, in which spatial modeling plays an important role, is critical to sustainable development. For more information about spatial analysis and modeling on tornoado, see 
[spatial trends in united states tornado frequency](https://www.nature.com/articles/s41612-018-0048-2), and [Spatial Redistribution of U.S. Tornado Activity between 1954 and 2013
](https://journals.ametsoc.org/view/journals/apme/55/8/jamc-d-15-0342.1.xml).

Of course, spatial modelling has many applications other than meteorology. Actually, spatial modeling such as smoothing and interpolation have "immense use in everything from anthropology to zoology and all the 'ologies' in-between". If we all agree that science is important for sustainable development, spatial modeling plays a fundamental role in these sciences.

---
<br/>
### How to Model Spatial-Temporal Phenomena?
An simple answer is: we use statistical models.
How to use statistical models? There is no simple answer. I will write a series of blogs to share what I am learning on this topic.
Why the model has to be statistical. 
>The primary reason comes from the uncertainty resulting from incomplete knowledge of the science and of the mechanisms driving a spatio-temporal phenomenon. In particular, statistical spatio-temporal models give us the ability to model components in a physical system that appear to be random and, even if they are not, the models are useful if they result in accurate and precise predictions. Such models introduce the notion of uncertainty, but they are able to do so without obscuring the salient trends or regularities of the underlying process (that are typically of primary interest).

> Take, for instance, the raindrops falling on a surface; to predict exactly where and when
each drop will fall would require an inconceivably complex, deterministic, meteorological model, incorporating air pressure, wind speed, water-droplet formation, and so on. A model of this sort at a large spatial scale is not only infeasible but also unnecessary for many purposes. By studying the temporal intensity of drops on a regular spatial grid, one can test for spatio-temporal interaction or look for dynamic changes in spatial intensity (given in units of “per area”) for each cell of the grid. The way in which the intensity evolves over time may reveal something about the driving mechanisms (e.g., wind vectors) and be useful for prediction, even though the exact location and time of each incident raindrop is uncertain.

I summarized the above statement as: it is neither necessary nor possible to observe complete data of the spatio-temporal phenomena.
Next, I will introduce three objectives of spatial modeling and two spatio-temporal statistical modeling types. 

[1]: https://spacetimewithr.org/Spatio-Temporal%20Statistics%20with%20R.pdf (Wikle, C. K., Zammit-Mangion, A., & Cressie, N. (2019). Spatio-Temporal Data in R. In Journal of Statistical Software.)
