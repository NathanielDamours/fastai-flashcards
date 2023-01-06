# 03_ethics

## Does ethics provide a list of "right answers"?

There is **no list of do's and dont's**. Ethics is complicated, and context-dependent. It involves the perspectives of many stakeholders. Ethics is a muscle that you have to develop and practice. In this chapter, our goal is to provide some signposts to help you on that journey.

## How can working with people of different backgrounds help when considering ethical questions?

Different people's backgrounds will help them to see things which may not be obvious to you. Working with a team is helpful for many "muscle building" activities, including this one.

## What was the role of IBM in Nazi Germany?

**IBM supplied the Nazis with data tabulation products necessary to track the extermination of Jews and other groups on a massive scale**. This was driven from the top of the company, with marketing to Hitler and his leadership team. Company President Thomas Watson personally approved the 1939 release of special IBM alphabetizing machines to help organize the deportation of Polish Jews. Hitler awarded Watson a special "Service to the Reich" medal in 1937.

But it also happened throughout the organization. IBM and its subsidiaries provided regular training and maintenance on-site at the concentration camps: printing off cards, configuring machines, and repairing them as they broke frequently. IBM set up categorizations on their punch card system for the way that each person was killed, which group they were assigned to, and the logistical information necessary to track them through the vast Holocaust system. IBM's code for Jews in the concentration camps was 8, where around 6,000,000 were killed. Its code for Romanis was 12 (they were labeled by the Nazis as "asocials", with over 300,000 killed in the  *Zigeunerlager* , or "Gypsy camp"). General executions were coded as 4, death in the gas chambers as 6.

## Why did the company and the workers participate as they did?

Because they were **making huge profits**.

Edwin Black, author of *IBM and the Holocaust*, said:
> To the blind technocrat, the means were more important than the ends. The destruction of the Jewish people became even less important because the invigorating nature of IBM's technical achievement was only heightened by the fantastical profits to be made at a time when bread lines stretched across the world.

## What was the role of the first person jailed in the Volkswagen diesel scandal?

It was one of the engineers, James Liang, who just did what he was told.

## What was the problem with a database of suspected gang members maintained by California law enforcement officials?

It was found to be **full of errors**, including 42 babies who had been added to the database when they were less than 1 year old (28 of whom were marked as "admitting to being gang members"). In this case, there was no process in place for correcting mistakes or removing people once they’d been added.

## Why did YouTube's recommendation algorithm recommend videos of partially clothed children to pedophiles?

Because of the **centrality of metrics** in driving a financially important system. Indeed. when an algorithm has a metric to optimise, it will do everything it can to optimise that number. This tends to lead to all kinds of edge cases, and humans interacting with a system will search for, find, and exploit these edge cases and feedback loops for their advantage.

## What are (3/7) problems with the centrality of metrics?

- Reliance on metrics can lead to a **narrow focus on measurable outcomes**, rather than broader goals or values. This can lead to a lack of attention to other important aspects of a situation or problem.

- Over-reliance on metrics can create pressure for people **to conform to predetermined goals or targets**, rather than encouraging creativity, innovation, or independent thinking.

- The use of metrics may create unintended consequences, such as **discrimination or bias**, if they are not carefully designed and implemented.

- The use of metrics may **create a sense of competition or rivalry**, rather than collaboration or cooperation.

- Metrics may **not accurately reflect the complexity or nuances of a situation**. They may oversimplify or obscure important factors or dynamics.

- Metrics may **not always be applicable or relevant in all contexts**, and may not capture the full range of factors that are important to consider in a given situation.

- Metrics may be **difficult to interpret or compare**, especially if they are not clearly defined or standardized.

[Source](https://chat.openai.com/chat)

## Why did Meetup.com not include gender in their recommendation system for tech meetups?

Because they were concerned that including gender in the recommendation algorithm would **create a self-reinforcing feedback loop where it would recommend Tech meetups mainly to men**, because Meetup had observed that men expressed more interest than women towards attending Tech meetups. To avoid this situation and continue to recommend Tech meetups to their users regardless of the gender, they simply decided to not include gender in the recommendation algorithm.

## What are the (6) types of bias in machine learning, according to Suresh and Guttag?

- Historical
- Measurement
- Aggregation
- Representation
- Deployment
- Evaluation

## What is historical bias?

A bias that our datasets and models inherit from the real world. People are biased, processes are biased and society in general is biased.

## What is measurement bias?

When we measure the wrong thing, incorporate the measurement inappropriately or measure it in the wrong way. An example is the stroke prediction model that includes information about if a person went to a doctor in it's prediction if a patient had a stroke.

## What is aggregation bias?

When data is aggregated to the extent where it is does not take the differences in the heterogeneous population of data into account. An example is that effectiveness of treatments in medicine for some diseases differs on gender and ethnicity, but where those parameters are not present in the training data as they have been "aggregated away"

## What is representation bias?

When the model emphasize some property of the data as it seemingly has the closest correlation with the prediction, even though that might not be the truth. An example is the gender property in the occupation prediction model where the model only predicted 11.6% of surgeons to be women whereas the real number was 14.6%.

## What is deployment bias?

When there is a disparity between the intended purpose of a model and how it is actually used. In other words, a model is designed for a purpose that is not achieved after deployment.
[Source](https://medium.com/unpackai/glimpse-of-different-types-of-bias-in-machine-learning-3e8767436aea)

## What is evaluation bias?

It happens during the evaluation or iteration process. It often arises when the testing or external target populations does not accurately represent the different segments of the user population. In addition, using inappropriate metrics for the intended use of the model can also lead to evaluation bias.

[Source](https://medium.com/unpackai/glimpse-of-different-types-of-bias-in-machine-learning-3e8767436aea)

## Give (2) examples of historical race bias in the US

- When doctors were shown identical files, they were much **less likely to recommend cardiac catherization (a helpful procedure) to Black patients.**
- An all-white jury was **16% more likely to convict a Black defendant than a white one**, but when a jury had at least one Black member, it convicted both at the same rate.

## Where are most images in Imagenet from?

**The US and other Western countries.** This leads to models trained on the ImageNet dataset performing worse for other countries and cultures that doesn't have as much representation in the dataset.

## How are machines and people different, in terms of their use for making decisions?

- People assume that algorithms are objective or/and error-free
- Algorithmic systems are:
  - more likely to be implemented with a no-appeals process in place
  - often used at scale
  - cheap

## Is disinformation the same as "fake news"?

No, it is not necessarily about getting someone to believe something false, but rather often **used to sow disharmony and uncertainty, and to get people to give up on seeking the truth**. To do that disinformation often contain exaggerations, seeds of truth or half-truths taken out of context rather than just "fake news". Also, disinformation has a history stretching back hundreds or even thousands of years.

## Why is disinformation through auto-generated text a particularly significant issue?

Due to the greatly **increased capability** provided by deep learning.

## What are the (5) ethical lenses described by the Markkula Center?

- **The rights approach**: which option best respects the rights of all who have a stake?
- **The justice approach**: which option treats people equally or proportionally?
- **The utilitarian approach**: which option will produce the most good and do the least harm?
- **The common good approach**: which option best serves the community as a whole, not just some members.
- **The virtue approach**: which option leads me to act as the sort of person I want to be?

The objective of looking through different ethical lenses when making a decision is to uncover concrete issues with the different options.

## When is policy an appropriate tool for addressing data ethics issues?

When it's likely that **design fixes, self regulation and technical approaches to addressing problems**, involving ethical uses of Machine Learning **are not working**. While such measures can be useful, they will not be sufficient to address the underlying problems that have led to our current state. For example, as long as it is incredibly profitable to create addictive technology, companies will continue to do so, regardless of whether this has the side effect of promoting conspiracy theories and polluting our information ecosystem. While individual designers may try to tweak product designs, we will not see substantial changes until the underlying profit incentives changes.

Because of the above it is almost certain that policies will have to be created by government to address these issues.
