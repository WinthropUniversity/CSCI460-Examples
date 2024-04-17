library(ggplot2)
library(reshape2)
library(dplyr)
library(RColorBrewer)

df <- read.csv("output.csv", header=T)

p <- ggplot(df, aes(x=n, y=firstHit)) + 
       geom_point(size=3,shape=21,fill="firebrick") +
       geom_smooth(method=lm, formula= y~x*log(x), size=1.5, color="firebrick") +
       theme_bw() + ylab("First Hitting Time")
print(p)
ggsave("output.pdf", width=10, height=7)

model = lm(firstHit ~ n*log(n), df)
print(summary(model))