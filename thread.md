Hi frens. 

In this thread i'll be detailing a little better fluxentropy and what our progress has been in the past couple of weeks, just so that we have a specific place to discuss this and for people to just jump in and contribute if they feel like it. 

Fluxentropy is a very simple idea: LLMs are not just generative models, but they're models of Language, and can hence be used to characterize text, somewhat similarly to what is currently done with encoder only-models. What specifically we think can be useful is the intrinsic ability of regular decoder-only models to represent next token distributions, and hence express uncertainty in with respect to the conditioning context, which is somewhat hard to do with regular encoder-only models, since their output is an actual vector representation of the context. 

Our objective is thus to create a set of tools that allow us to use models as support to other mechanism that are required in LLMs. An example (and our main current target of this) is curriculum learning: as you know, models are currently pretrained by randomly sampling strings from large, mostly unordered datasets. Our aim is to provide a scheduler that allows the model to learn from samples that are sorted with respect to a certain characteristic, which is for now entropy. Apart from its potential to increase training performance, this method also serves the important objective of giving us a set of quantitative proving grounds for the expressivity of model-predicted entropy, varentropy and other characterizations of the context. For example, we could find that some specific combination or normalization of entropy and varentropy are better sorters of datasets (leading to better pretraining dynamics) than how we currently work with them.

Before doing so, we setup a very basic set of experiments and code to have a few reality checks. 

First thing we did was actually run a set of models over a set of diverse prompts. this gave us a feeling of relationship between model families when it comes to their evaluation of entropy in the prompts.

First we plot a sorted version of the prompts with respect to their entropy value. As you can see, variance in models seems rather fixed, but entropy in general has different magnitudes. here the ordering by smollm 1.7 is taken as baseline to order the x axis, which corresponds to the individual prompts.

![](https://cdn.discordapp.com/attachments/1299312352709185557/1306006178223882300/image.png?ex=6744ea89&is=67439909&hm=e3df3129a53ad8d26f2d463dae3a27c311a03ff25dafe06de0cc55099d201086&=)

Next plot is similar, but plotting the ranks in the sorting depending on the model. As you can see, models seem to agree much more on extreme values of entropy, while more average values are more spread out. Highest entropy texts are the most aligned across models.

![](https://cdn.discordapp.com/attachments/1299312352709185557/1306006221580533801/image.png?ex=6744ea93&is=67439913&hm=019fccd7f74d15b9340816d1ded32afb17ab92282536a1cda26869bd489d2b29&=)
This exercise can actually help us compare models with respect to one another. One approach could be what we did here, which is pretending each individual prompt is an element of a vector, calculating euclidean distance. 

![distance_matrix.png](https://github.com/SinatrasC/fluxentropy/blob/current/sanity_checks/plots/distance_matrix.png?raw=true)
Somewhat similarly, this can be mapped to principal components and plot as we see here. It's interesting to notice that the first principal components explains >85% of the variance, and the second roughly 2%. 

![pca_models.png](https://github.com/SinatrasC/fluxentropy/blob/current/sanity_checks/plots/pca_models.png?raw=true)
Here we can somewhat notice some clustering in model families. But we can also see that size is pretty important.

We then moved to trying to predict model performance over an eval using entropy alone. This is crucial to try and see how strong of a signal we can expect of entropy. Our first (and only) try was with Hellaswag:
as we can see, the distributions do indeed seem to be distinct to some level, from plotting alone.

![distribution_entropy_only.png](https://github.com/SinatrasC/fluxentropy/blob/current/eval_entropy/output/plots/distribution_entropy_only.png?raw=true)

This can actually be modeled slightly better: remember, hellswag test for correctness in completion from very short prompts. We fit a logistic regression model and find slight, statistically significant predictive power for token-wise-aggregated entropy in distinguishing the two:

![](https://cdn.discordapp.com/attachments/1299312352709185557/1306740748141723679/image.png?ex=67444ae8&is=6742f968&hm=079231c46a005c334575c08308d7a3a194d19cee1c3ca2b5209d5399c9469ddc&=)

As you can see, though, the signal is very weak. This is probably due to the fact that the hellaswag signal itself is really sparse. probably an eval giving more quantitative and nuanced results could be fit better.

![](https://cdn.discordapp.com/attachments/1299312352709185557/1306740609125585007/image.png?ex=67444ac7&is=6742f947&hm=3acb78f27dcf531b5b6e344133b41e706d163941e2ea2e11ddf7195d9163915c&=)

Specifically, we also repeated this same experiments under a few regimes:

-entropy only aggregated token by token
-entropy only last token only
-entropy and varentropy aggregated token by token
-entropy and varentropy last token only

without showing all of the results, as it turns out entropy (or varentropy, but still by itself) is the only statistically significant variable under these conditions. Even using both entropy and varentropy (hence a bi-variate model), we get a still significant model as a whole, but its coefficients are not:

![](https://cdn.discordapp.com/attachments/1299312352709185557/1307088988238254100/image.png?ex=6744e67b&is=674394fb&hm=a971a988cf2c878ec3f923ece1880d2b8e3eab94f1264a7ef004da4c94da4e7f&=)

Here we used square root of varentropy, but the same happens for regular varentropy. As you can see, logits' p-value is still <0.05. 

interestingly, last-token-only seems a worse predictor than aggregate. 
You can find all of the stats in the repo, but here are examples of distributions for last-token-only:
![](https://cdn.discordapp.com/attachments/1299312352709185557/1307730921096740934/image.png?ex=67449954&is=674347d4&hm=319080136b977886e1ecb45ce680b5e16f8b3f83f692c200e5fdb9c112075841&=)

![](https://cdn.discordapp.com/attachments/1299312352709185557/1307730921796927488/image.png?ex=67449954&is=674347d4&hm=5c8c83b40dbbdf07397c8deef4bd9bd9cedbc98bc88f31ae4bacb7addd464f12&=)
This might have to do with hellaswag being very short prompts. 
Also, we found that entropy and varentropy correlated in different directions when aggregate and last token. The following is a correlation matrix from last-token-only distributions (here we had kept top-100 logprobs, so we're also re-calculating metrics as a reality check)
![](https://cdn.discordapp.com/attachments/1299312352709185557/1307730747557150731/image.png?ex=6744992a&is=674347aa&hm=69ffc845817cd3844e6b8884c8caac97a0eac42063d965fe4679270e8ee09711&=)
while this is a correlation matrix from aggregated metrics over all of the tokens in the prompt: ![](https://cdn.discordapp.com/attachments/1299312352709185557/1307093392047738981/image.png?ex=6744ea95&is=67439915&hm=d9675dd55a974e13988a0a359994023db4cbf65b78625a3e7e329d5090f49c5d&=)
here the important part is: i expect varentropy to be negative correlated with entropy when it comes to representing confusion. This plot by itself doens't prove or disprove this though. 

This was of course a sidequest, but i think it may give us some insight into how we can quantitatively test these metrics to better gauge how to mix them to express model confusion effectively. 
Currently, we're working on hacking this to the nanoGPT speedrun pipeline, specifically by setting a number of entropy bins and scheduling them in increasing order.
We are currenly bottlenecked by compute and time: we have in place code to sort fineweb10B, we just have to run it, store somewhere, and then try and train nanoGPT models. 
Here you can find a reference of the approach we are trying to take with respect to the curriculum learning: [https://discord.com/channels/@me/1299312352709185557/1307777982483529758](https://github.com/kamel-yamani/CurriculumLearningForSmallCodeLanguageModels)
This in general could at least open a few interesting ideas: 
- does entropy work? 
- if not, what works? can it be made a quantifiable measure to find some confusion metric that we can use for the sampler?
- what scheduling works best? finer or coarser bins? what if bins were not only increasing, but had a specific schedule to them? what about it being periodic? 

this is the repo: https://github.com/SinatrasC/fluxentropy/tree/current , feel free to discuss, give us feedback, take a look and contribute!
