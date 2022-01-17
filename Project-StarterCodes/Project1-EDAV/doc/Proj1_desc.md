## LEAP Education 
#### Spring 2022
### Climate Predication Challenges 
### Project 1: Exploring Association bewteen Hurricane Tracks and Climate Change
### --- A "*Data Story*" using Jupyter Notebook on Google Colab

<img src="../figs/tracks-2020example.jpeg" width="400">

It remains relatively uncertain how climate change affects extreme weather such as hurricanes. It has been suggested that the changing climate may lead to warmer sea surface temperatures, sea level rises, and shifts of the areas affected by hurricanes. Hurricanes, as both directly observed behaviors of the earth system and extreme weather that has substantial impacts on human lives and society, are, therefore, of great interest to climate scientists. 

The goal of this project is to look deeper into the patterns and characteristics of hurricane tracks from seasons that date back as far as 1850s and how they associate with climate forcing factors. Applying data mining, statistical analysis and visualization, students should derive interesting findings in this collection of hurricane tracks, search the climate research literature for potential hypotheses, identify other useful data sources, and write a "data story" that can be shared with a **general audience**. 

### Datasets

+ International Best Track Archive for Climate Stewardship [(IBTrACS)](https://www.ncdc.noaa.gov/ibtracs/) from NOAA National Centers for Environmental Information. 
+ The data will be downloaded by the starter codes provided. 

### Challenge 

In this project you will carry out an **exploratory data analysis (EDA)** of hurricane tracks and write a reproducible data analysis *notebook* on interesting findings (i.e., a *data story*).

You are tasked to explore the data, driven by the climate research literture and interests among your team mates, using tools from data mining, statistical analysis and visualization, etc, all available in `Python` and create `Python/Jupyter` Notebook in [Google Colab](https://www.youtube.com/watch?v=inN8seMm7UI). Your notebook should be in the form of a `data story`, with both codes and text that describe the motivation of your analysis, the steps of your finalized analysis and discussions on interesting trends and patterns identified in your analysis. Your notebook should be organized in a logical order and contain only the codes and discussions of your finalized analysis.   

### Project organization

The project starter codes can be opened by following the link in its [GitHub copy](https://github.com/leap-stc/LEAPCourse-Climate-Pred-Challenges/blob/main/Project-StarterCodes/Project1-EDAV/lib/Project1-Starter.ipynb), which is also linked in courseworks. 

All the codes in this *starter codes* notebook can be modified to implement your research ideas.

To start, everyone,

+ create in your google drive a folder for this course.

The team leader of each team should

+ create in the course folder a folder for project 1.
This folder can be used to share notes and codes.
+ share project 1 folder with all team members.
	+ Team members should all create a shortcut to this shared project folder in their own folder for LEAP CPC.
+ go to "File/Save A Copy in Drive/" (upper left) and save a copy for your team in the project 1 folder that was just created.

Final product of this project is a Google Colab notebook that the team produce together, which will be presented on Feburary 8th, 2022. 
 
#### Suggested workflow
This is a relatively shorter project of the semester. We have about three weeks of working time. In the starter codes, we provide you basic basic data processing steps to show you how to get started. 

1. [wk1] Week 1 is the **data processing and mining** week. Read data description, **project requirement**, browse data and studies the R notebooks in the starter codes, and think about what to do and try out different tools you find related to this task.
2. [wk1] Try out ideas on a **subset** of the data set to get a sense of computational burden of this project. 
3. [wk2] Explore data for interesting trends and start writing your data story. 

#### Submission
You should produce an R notebook (rmd and html files) in your GitHub project folder, where you should write a story or a blog post on "How Did Americans Vote" based on your data analysis. Your story, especially *main takeways* should be **supported by** your results and appropriate visualization. 

Your story should NOT be a laundry list of all analyses you have tried on the data or how you solved a technical issue in your analysis, no matter how fascinating that might be. 

#### Repository requirement

The final repo should be under our class github organization (TZStatsADS) and be organized according to the structure of the starter codes. 

```
proj/
├──data/
├──doc/
├──figs/
├──lib/
├──output/
├── README
```
- The `data` folder contains the raw data of this project. These data should NOT be processed inside this folder. Processed data should be saved to `output` folder. This is to ensure that the raw data will not be altered. 
- The `doc` folder should have documentations for this project, presentation files and other supporting materials. 
- The `figs` folder contains figure files produced during the project and running of the codes. 
- The `lib` folder (sometimes called `dev`) contain computation codes for your data analysis. Make sure your README.md is informative about what are the programs found in this folder. 
- The `output` folder is the holding place for intermediate and final computational results.

The root README.md should contain your name and an abstract of your findings. 

### Useful resources

##### R pakcages
* R [tidyverse](https://www.tidyverse.org/) packages
* R [DT](http://www.htmlwidgets.org/showcase_datatables.html) package
* R [tibble](https://cran.r-project.org/web/packages/tibble/vignettes/tibble.html)
* [Rcharts](https://www.r-graph-gallery.com/interactive-charts.html), quick interactive plots
* [htmlwidgets](http://www.htmlwidgets.org/), javascript library adaptation in R. 

##### Project tools
* A brief [guide](http://rogerdudler.github.io/git-guide/) to git.
* Putting your project on [GitHub](https://guides.github.com/introduction/getting-your-project-on-github/).

##### Examples
+ [FiveThirtyEight: Voters Who Think The Economy Is The Country’s Biggest Problem Are Pretty Trumpy. That Might Not Help Him Much.](https://fivethirtyeight.com/features/voters-who-think-the-economy-is-the-countrys-biggest-problem-are-pretty-trumpy-that-might-not-help-him-much/)
+ [Blog: Republicans trust the government more than Democrats do under their own presidents](https://blogs.lse.ac.uk/usappblog/2019/10/31/republicans-trust-the-government-more-than-democrats-do-under-their-own-presidents/)
+ [Paper: “Like They’ve Never, Ever Seen in This Country”? Political Interest and Voter Engagement in 2016](https://academic.oup.com/poq/article/82/S1/822/4944388)
+ [A good "data story"](https://drhagen.com/blog/the-missing-11th-of-the-month/)

##### Tutorials

For this project we will give **tutorials** and give comments on:

- GitHub
- R notebook
