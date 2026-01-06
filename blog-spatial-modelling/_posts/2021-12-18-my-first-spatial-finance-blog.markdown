---
layout: post-wide
title:  "What is Spatial Finance?"
date:   2021-12-30 15:41:32 +0800
category: Spatial Modeling
author: Hank Li
---

Some websites state that the biggest opportunity over the next decade is AI. I do not buy it. I think the biggest opportunity lies on what people want and what problems they have to solve.
Well then, what is the biggest problem we will face in the next a few decades? I have no idea. I only know that everyone wants to love and be loved:)
Nothing else matters... unless the place where we enjoy love is in danger. The blog is about spatial finance. If I say something non sense, it might be because that I rencently watched the movie `Don't Look Up`:)

### What is Spatial Finance?

Spatial Finance is a geospatial-driven approach designed to provide ESG relevant insights into specific commercial assets, companies or portfolios [[1]](https://www.wwf.org.uk/sites/default/files/2020-12/Spatial%20Finance_%20Challenges%20and%20Opportunities_Final.pdf){:target="_blank"}.

According to [Spatail Finance Initiative](https://www.cgfi.ac.uk/spatial-finance-initiative/){:target="_blank"}:
> Spatial finance is the integration of geospatial data and analysis into financial theory and practice.
> Earth observation and remote sensing combined with machine learning have the potential to transform the availability of information in our financial system. It will allow financial markets to better measure and manage climate-related risks, as well as a vast range of other factors that affect risk and return in different asset classes.

If you haven't heard another relevant term 'spatial econometrics', just ignore the following sentences of this paragraph.
Otherwise, it is worth noting the difference between spatial finance and spatial econometrics.
`Spatial econometrics` refers to the study of a set of special regression models, of which the data observations are not truly independent.
`Spatial econometrics` incorporate spatial auto-correlation and neighborhood effects into the regression models in order to better fit the data.
`Spatial finance`, however, is neither a new model nor a new modeling methodology. It is just using spatial data science in finance to provide ESG relevant insights.

Spatial finance is also foundamentally different from other `Fintech` such as blockchain and cryptocurrency, stock-trading apps, personal finance, crowd funding, consumer banking, etc.
`Fintech` focuses on financial activities such as transfering money, depositing a check, applying for a credit card, raising money for a business startup, or managing investments with smartphones.
However, `spatial fiance` is essentially to use remote sensing and other earth observation technology to get up-to-date, high resolution environmental or climate datasets covering metrics across a wide portfolio to analyze commercial asset data.

**The core of spatial finance** is to *develope and maintain commerical asset datasets as well as providing a platform for exchanging these datasets*.
Thus, the following will introduce what commerical asset datasets we have and what geospatial technology can be used to facilitate dataset development.

### Five Tiers of Spatial Finance Datasets?
[WWF spatial finance whitepaper](https://www.wwf.org.uk/sites/default/files/2020-12/Spatial%20Finance_%20Challenges%20and%20Opportunities_Final.pdf){:target="_blank"} proposed five tiers of spatial finance data:
5. Tier 4: Sub-asset level. For example, air pollution monitors, smart power meters and industry specific measurements. These data are often recorded within the asset at a high temporal frequency.
4. Tier 3: Asset level. Cover the full extent of a specific asset such as water risk of the site, nearby deforestation, carbon emissions, etc.
3. Tier 2: Company or parent company level. Aggregate Tier 3 and 4 data to provide parent company level analysis. Supply chain assessments is important at this tier, as the environmental impact of a parent company is determined by their supply chains rather than the location of their physical assets such as headquarters.
2. Tier 1: Portfolio level. Further simplify tier 2 data and provide results at a portfolio level.
1. Tier 0: Country level. Aggregate all tiers data at the country level.

Some open datasets include:
1. [GeoAsset project](https://www.cgfi.ac.uk/spatial-finance-initiative/geoasset-project/){:target="_blank"}: Open asset-level data for cement and steel sectors by Spatial Finance Initiative.
2. [Global Power Plant Dataset](https://datasets.wri.org/dataset/globalpowerplantdatabase){:target="_blank"} developed by World Resource Institute (WRI) and Google.
3. [The Global Tailings Portal](https://tailing.grida.no/){:target="_blank"}: An open database with detailed information on global mine tailings dams.

Just make notes for myself. Here are a few [Open Source Geospatial Data Sets](https://geospatialawarenesshub.com/blog/top-open-source-geospatial-data-sets/){:target="_blank"}.

### What technology is used to build and maintain the dataset?

Machine learning and AI (Deep learning related algorithms) will play a vital role here. For example, Spatial Finance Initiative shared how these databases were constructed.
> We have used a mixture of methods to develop these databases. These methods included the use of remote sensing imagery, machine learning, crowdsourcing and desk research to identify and characterise cement and iron & steel production facilities globally. Initially, desk research was undertaken to develop training data and to link identified assets to its owner. After this initial desk research, machine learning models were developed and applied to remote sensing imagery to identify new plants that had not already been identified. This identification strategy was also complemented by crowdsourcing efforts to validate and annotate the identified facilities, which are used for the identification of plant and production type as well for the purposes of estimating the production capacity for cement plants.

Some attribute information such as capacity of a cement asset has to be estimated or found from disclosed infromation.
> For most cement assets capacity was obtained from disclosed information by the companies or by an industry association. For some cement assets this capacity information was estimated using the dimensions of the plant and the number and dimensions of the kilns located at that plant. For those plants with modelled capacity the capacity source within the database is reported as “Estimated.” For all iron & steel assets the capacity was obtained from the capacity source disclosed within the database.

### How Spatial Finance is associated with spatial modeling and spatial optimization?
<div class="mermaid">
flowchart LR;
    subgraph ABC[Geospatial Technology];
      direction TB;
      A[<b>Spatial Finance</b><br/>Use Machine Learning and GeoAI <br/>to build the asset datasets]-->B[<b>Spatial Modeling</b><br/>Use Statistical models<br/>to gain insights from these datasets];
      A-->C[<b>Spatial Optimization</b><br/>Use Optimization methods<br/>to find the best solution];
    end
    ABC-->D[Sustainable Development]
</div>

Spatial finance plays an foundamental role in providing up-to-date vital commerical asset datasets, based on which spatial modeling and spatial optimization are used to facilitate insights retriving and ESG report generation. Metaverse can be used as a sandbox for simulation, evalution, and education.

*People may be willing to pay more to get climate-friendly products*(see [here](https://theconversation.com/climate-explained-are-consumers-willing-to-pay-more-for-climate-friendly-products-146757){:target="_blank"}). People are willing to pay to make the world a better place. It is positive. What a wonderful world!

