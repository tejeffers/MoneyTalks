# MoneyTalk$
### _Predicting US Federal Campaign Contributions using Demographic Data_

#### The Problem:
Political donations are terribly important.

* First, it's expensive to run for political office. Experts estimate that running for (and winning) the presidency in 2016 cost somewhere between 3 and 5 billion dollars. 
* Second, donations by individuals are an act of investment. People are essentially buying in to the process, and probably expect something in return.

One challenge is that **the majority of money comes from the minority of donors**, obfuscating exactly who the campaign is meant to represent.

![Campaign Contributions CDF](https://github.com/tejeffers/MoneyTalks/blob/master/02_images/contributions.png)


For my Insight Data Science Fellowship project, I built an interactive web application that uses US Census demographics to predict donations in a geographical area, thus helping campaign fundraisers answer two critical questions:

1. Who is the candidate supposed to represent?
2. Where should they target their efforts to raise more money?

#### The Approach:

To predict campaign contributions in a zip code, I'm using:

* Individual Campaign Contributions, aggregated by the Federal Elections Committee, 
	and
* US Demographic Data from the American Community Survey.

After normalizing and correcting for missing and skewed data, I built a 135-feature Gradient Boosting Regression model to predict the total campaign contributions in a zip code using the demographic features of that area.
![The Approach](https://github.com/tejeffers/MoneyTalks/blob/master/02_images/approach.png)
In addition, I used hierarchical agglomerative clustering to generate demographic profiles that I could use to better understand contribution behavior across different demographic clusters.

Together, this yields a predictive model for targeting and visualizing underperforming zip codes. Take a look at the online [MONEYTALK$ DASHBOARD](http://tessjeffers.com)!

#### The Results:

<img align="right" src="https://github.com/tejeffers/MoneyTalks/blob/master/02_images/GBR_model_zipcodes.png">

For each zip code in the US, I calculate the expected total $ it should raise. If perfectly accurate, all zip codes (blue dots) should fall on the 45 degree line (gold line). Of course, my model isn't perfect, and in fact these outliers represent an opportunity. Take for example the zipcode '10024', the Upper West Side in Manhattan. My model predicts the UWS should raise ~ $1,700,000. In actuality, this zip code alone raised over $17,000,000. Knowing the location of these 'over-performing' zip codes could be useful for planning high-budget donation events, like $10,000 plate dinners, galas, etc.

On the other end of the spectrum, 'under-performing' zip codes represent locations where, for whatever reason, people have not donated to the capacity that the model predicts they should. These zip codes represent an opportunity for a candidate to refine their strategy, send out additional flyers, or otherwise attempt better messaging to connect with the demographics in that area.


